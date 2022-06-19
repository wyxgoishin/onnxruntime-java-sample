package util;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class OnnxTensorUtil {
    private static final Logger logger = LoggerFactory.getLogger(OnnxTensorUtil.class);
    public static final int[] torchTensorDimOrders = new int[]{2, 0, 1};
    public static final int[] defaultTensorDimOrders = new int[]{0, 1, 2};

    /*
     * Author wyxgoishin
     * Description transform a single Mat (H * W * C) to Tensor (1 * H * W * C)
     * Date 2022/6/15 16:11
     * Param [env, mat]
     * return ai.onnxruntime.OnnxTensor
     **/
    public static OnnxTensor createTensorFromImageMat(OrtEnvironment env, Mat mat, int[] dimOrders){
        return createTensorFromImageMats(env, List.of(mat), dimOrders);
    }

    /*
     * Author wyxgoishin
     * Description Transform list of Mat (H * W * C) to Tensor (B * H * W * C)
     * Date 2022/6/15 16:11
     * Param [env, mats]
     * return ai.onnxruntime.OnnxTensor
     **/
    public static OnnxTensor createTensorFromImageMats(OrtEnvironment env, List<Mat> mats, int[] dimOrders){
        // Ensure all the input mats share same shape (h, w, c)
        int[] shape = checkAndGetMatShape(mats);
        if(shape == null){
            return null;
        }

        int batch = mats.size();
        int height = shape[0];
        int width = shape[1];
        int channels = shape[2];
        int depth = shape[3];
        int[] indexes = new int[3];

        OnnxTensor tensor = null;

        long cap = (long) batch * height * width * channels;
        // MaxSize of java.nio.Buffer is Integer.MAX_VALUE
        if(cap > Integer.MAX_VALUE){
            // Construct OnnxTensor from Array, which is slower than Buffer
            logger.warn("Excessive Mat element size {}, will use array to construct tensor", cap);

            /*
             * Create a buffer array for constructing OnnxTensor and its last three dims are determined by indexing
             * from shape with given dimOrders
             **/
            float[][][][] bufArr = new float[batch][shape[dimOrders[0]]][shape[dimOrders[1]]][shape[dimOrders[2]]];
            // Create buffer array for reading from Mat and its element type is determined by Mat depth
            Object buf = initBuf(channels, depth);
            for(int b = 0; b < batch; b++){
                Mat mat = mats.get(b);
                for(int h = 0; h < height; h++){
                    indexes[0] = h;
                    for(int w = 0; w < width; w++){
                        indexes[1] = w;
                        readMatToBuf(mat, buf, depth, h, w);
                        for(int ch = 0; ch < channels; ch++){
                            indexes[2] = ch;
                            // Read value from buffer array and transform the value into real unsigned value if needed
                            float val = readValueFromSignedBuf(buf, depth, ch);
                            bufArr[b][indexes[dimOrders[0]]][indexes[dimOrders[0]]][indexes[dimOrders[0]]] = val;
                        }
                    }
                }
            }

            try{
                tensor = OnnxTensor.createTensor(env, bufArr);
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }else{
            // Construct OnnxTensor from Buffer, default
            /* calculate the factors for indexing buffer in row-major manner and the dim values are determined
             * by indexing from shape with given dimOrders
             **/
            int[] factors = getFactors(shape[dimOrders[0]], shape[dimOrders[1]], shape[dimOrders[2]]);
            FloatBuffer floatBuffer = FloatBuffer.allocate((int) cap);
            /*
             * Create a buffer array for constructing OnnxTensor and its last three dims are determined by indexing
             * from shape with given dimOrders
             **/
            Object buf = initBuf(channels, depth);
            for(int b = 0; b < batch; b++){
                Mat mat = mats.get(b);
                for(int h = 0; h < height; h++){
                    indexes[0] = h;
                    for(int w = 0; w < width; w++){
                        indexes[1] = w;
                        readMatToBuf(mat, buf, depth, h, w);
                        for(int ch = 0; ch < channels; ch++){
                            indexes[2] = ch;
                            // Read value from buffer array and transform the value into real unsigned value if needed
                            float val = readValueFromSignedBuf(buf, depth, ch);
                            // Calculate index in row-major manner
                            int index = b * factors[0] + indexes[dimOrders[0]] * factors[1] + indexes[dimOrders[1]] * factors[2] + indexes[dimOrders[2]];
                            floatBuffer.put(index, val);
                        }
                    }
                }
            }

            try{
                // Create shape of tensor and its last three dims are determined by indexing shape with given dimOrders
                long[] tensorShape = new long[]{batch, shape[dimOrders[0]], shape[dimOrders[1]], shape[dimOrders[2]]};
                tensor = OnnxTensor.createTensor(env, floatBuffer, tensorShape);
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }

        return tensor;
    }

    /*
     * @Author wyxgoishin
     * @Description Create a OnnxTensor of Pytorch style (1 * C * H * W) from single image Mats (H * W * C)
     * @Date 2022/6/13 13:41
     * @Param
     * @Return OnnxTensor
     **/
    public static OnnxTensor createTorchTensorFromImageMat(OrtEnvironment env, Mat mat){
        return createTensorFromImageMat(env, mat, torchTensorDimOrders);
    }

    /*
     * @Author wyxgoishin
     * @Description Create a OnnxTensor of Pytorch style (B * C * H * W) from image Mats (H * W * C)
     * @Date 2022/6/13 13:38
     * @Param
     * @Return OnnxTensor
     **/
    public static OnnxTensor createTorchTensorFromImageMats(OrtEnvironment env, List<Mat> mats) {
        return createTensorFromImageMats(env, mats, torchTensorDimOrders);
    }

    /*
     * @Author wyxgoishin
     * @Description Transform a Pytorch style tensor (1 * C * H * W) to a 3-channel Mat, ensure C <= 3
     * @Date 2022/6/13 14:02
     * @Param
     * @Return
     **/
    public static Mat createImageMatFromTensor(OnnxTensor tensor){
        return createImageMatsFromTensor(tensor).get(0);
    }

    /*
     * @Author wyxgoishin
     * @Description Transform a Pytorch style  tensor (B * C * H * W) to list of 3-channel Mats, ensure C <= 3
     * @Date 2022/6/13 13:44
     * @Param
     * @Return
     **/
    public static List<Mat> createImageMatsFromTensor(OnnxTensor tensor){
        long[] shape = tensor.getInfo().getShape();
        int channels = (int) shape[1];
        if(channels > 3){
            logger.error("Expected channel of given Tensor to be less than 3, got {} instead", channels);
            throw new RuntimeException();
        }

        int batch = (int) shape[0];
        int height = (int) shape[2];
        int width = (int) shape[3];
        int type = CvType.CV_32FC3;

        List<Mat> mats = new ArrayList<>(batch);
        long cap = (long) batch * channels * height * width;
        // MaxSize of java.nio.Buffer is Integer.MAX_VALUE
        if(cap > Integer.MAX_VALUE){
            // Construct Mats from Array, which is slower
            logger.warn("Excessive tensor element num {}, will use array to construct mats", cap);

            float[][][][] bufArr;
            try{
                bufArr = (float[][][][]) tensor.getValue();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }

            float[] buf = new float[3];
            for(int b = 0; b < batch; b++){
                Mat mat = Mat.zeros(height, width, type);
                for(int h = 0; h < height; h++){
                    for(int w = 0; w < width; w++){
                        for(int ch = 0; ch < channels; ch++){
                            buf[ch] = bufArr[b][ch][h][w];
                        }
                        mat.put(h, w, buf);
                    }
                }
                mats.add(mat);
            }
        }else{
            // Construct Mats from Buffer, default
            /* calculate the factors for indexing buffer in row-major manner and the dim values are determined
             * by indexing from shape with given dimOrders
             **/
            int[] factors = getFactors(channels, height, width);

            FloatBuffer floatBuffer = tensor.getFloatBuffer();
            float[] buf = new float[3];
            for(int b = 0; b < batch; b++){
                Mat mat = Mat.zeros(height, width, type);
                for(int h = 0; h < height; h++){
                    for(int w = 0; w < width; w++){
                        for(int ch = 0; ch < channels; ch++){
                            // Calculate index in row-major manner
                            int index = b * factors[0] + ch * factors[1] + h * factors[2] + w;
                            buf[ch] = floatBuffer.get(index);
                        }
                        mat.put(h, w, buf);
                    }
                }
                mats.add(mat);
            }
        }

        return mats;
    }

    /*
     * @Author wyxgoishin
     * @Description Transform a Pytorch style optical flow tensor (1 * 2 * H * W) to a KITTI format Mats
     * @Date 2022/6/13 14:14
     * @Param
     * @Return
     **/
    public static Mat flowTensorToKittiMat(OnnxTensor tensor, int[] dimOrders){
        return flowTensorToKittiMats(tensor, dimOrders).get(0);
    }

    /*
     * @Author wyxgoishin
     * @Description Transform a Pytorch style optical flow tensor (B * 2 * H * W) to list of KITTI format Mats
     * @Date 2022/6/13 14:04
     * @Param
     * @Return
     **/
    public static List<Mat> flowTensorToKittiMats(OnnxTensor tensor, int[] dimOrders){
        long[] shape = tensor.getInfo().getShape();
        int channels = (int) shape[dimOrders[2] + 1];
        if(channels != 2){
            logger.error("Expected channel of Flow Tensor to be 2, got {} instead", channels);
            throw new RuntimeException();
        }
        int batch = (int) shape[0];
        int height = (int) shape[dimOrders[0] + 1];
        int width = (int) shape[dimOrders[1] + 1];
        int type = CvType.CV_16UC3;
        int[] indexes = new int[3];

        List<Mat> mats = new ArrayList<>(batch);
        long cap = (long) batch * channels * height * width;
        // MaxSize of java.nio.Buffer is Integer.MAX_VALUE
        if(cap > Integer.MAX_VALUE){
            // Construct Mats from Array, which is slower
            logger.warn("Excessive tensor element num {}, will use array to construct mats", cap);

            float[][][][] bufArr;
            try{
                bufArr = (float[][][][]) tensor.getValue();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }

            float[] floatBuf = new float[3];
            // kitti-format image is stored in uint16
            short[] shortBuf = new short[3];
            for(int b = 0; b < batch; b++){
                Mat mat = Mat.zeros(height, width, type);
                for(int h = 0; h < height; h++){
                    indexes[0] = h;
                    for(int w = 0; w < width; w++){
                        indexes[1] = w;
                        for(int ch = 0; ch < 2; ch++){
                            indexes[2] = ch;
                            // kitti-flow : f(x) = x * 64 + 2 ** 15, in uint16 format
                            floatBuf[ch] = (float) Math.min(bufArr[b][indexes[dimOrders[0]]][indexes[dimOrders[1]]][indexes[dimOrders[2]]] * 64.0 + 32768.0, 65535.0);
                            floatBuf[ch] = Math.max(-65535, floatBuf[ch]);
                            // change signed value to unsigned one
                            floatBuf[ch] = floatBuf[ch] > 32767 ? floatBuf[ch] - 65536: floatBuf[ch];
                            shortBuf[ch] = (short) Math.round(floatBuf[ch]);
                        }
                        // three channel value for kitti-flow : 1, v, u
                        shortBuf[2] = shortBuf[0];
                        shortBuf[0] = 1;
                        mat.put(h, w, shortBuf);
                    }
                }
                mats.add(mat);
            }
        }else{
            // Construct Mats from Buffer, default
            /* calculate the factors for indexing buffer in row-major manner and the dim values are determined
             * by indexing from shape with given dimOrders
             **/
            int[] factors = getFactors((int) shape[1], (int) shape[2], (int) shape[3]);

            FloatBuffer floatBuffer = tensor.getFloatBuffer();
            float[] floatBuf = new float[3];
            // kitti-format image is stored in uint16
            short[] shortBuf = new short[3];
            for(int b = 0; b < batch; b++){
                Mat mat = Mat.zeros(height, width, type);
                for(int h = 0; h < height; h++){
                    indexes[0] = h;
                    for(int w = 0; w < width; w++){
                        indexes[1] = w;
                        for(int ch = 0; ch < channels; ch++){
                            indexes[2] = ch;
                            int index = b * factors[0] + indexes[dimOrders[0]] * factors[1] + indexes[dimOrders[1]] * factors[2] + indexes[dimOrders[2]];
                            // kitti-flow : f(x) = x * 64 + 2 ** 15, in uint16 format
                            floatBuf[ch] = (float) Math.min(floatBuffer.get(index) * 64.0 + 32768.0, 65535.0);
                            floatBuf[ch] = Math.max(-65535, floatBuf[ch]);
                            // change signed value to unsigned one
                            floatBuf[ch] = floatBuf[ch] > 32767 ? floatBuf[ch] - 65536: floatBuf[ch];
                            shortBuf[ch] = (short) Math.round(floatBuf[ch]);
                        }
                        // three channel value for kitti-flow : 1, v, u
                        shortBuf[2] = shortBuf[0];
                        shortBuf[0] = 1;
                        mat.put(h, w, shortBuf);
                    }
                }
                mats.add(mat);
            }
        }

        return mats;
    }

    /*
     * @Author wyxgoishin
     * @Description Init a buf array determined by depth and channel for reading Mat
     * @Date 2022/6/13 14:58
     * @Param
     * @Return
     **/
    private static Object initBuf(int channels, int depth) {
        Object buf = null;
        switch (depth){
            case CvType.CV_8S: case CvType.CV_8U: buf = new byte[channels]; break;
            case CvType.CV_16S: case CvType.CV_16U: buf = new short[channels]; break;
            case CvType.CV_32S: buf = new int[channels]; break;
            case CvType.CV_32F: buf = new float[channels]; break;
            default: throw new RuntimeException(String.format("Unimplemented Mat depth %d", depth));
        }
        return buf;
    }

    /*
     * @Author wyxgoishin
     * @Description Read Mat values at (h, w) to an Object buf
     * @Date 2022/6/13 14:58
     * @Param
     * @Return
     **/
    private static void readMatToBuf(Mat mat, Object buf, int depth, int h, int w){
        switch (depth){
            case CvType.CV_8S: case CvType.CV_8U: mat.get(h, w, (byte[]) buf); break;
            case CvType.CV_16S: case CvType.CV_16U: mat.get(h, w, (short[]) buf); break;
            case CvType.CV_32S: mat.get(h, w, (int[]) buf); break;
            case CvType.CV_32F: mat.get(h, w, (float[]) buf); break;
            default: throw new RuntimeException(String.format("Unimplemented Mat depth %d", depth));
        }
    }

    /*
     * @Author wyxgoishin
     * @Description Read value at index from a buf and convert value from signed to unsigned
     * @Date 2022/6/13 14:59
     * @Param
     * @Return
     **/
    private static float readValueFromSignedBuf(Object buf, int depth, int idx){
        float val = 0;
        switch (depth){
            case CvType.CV_8U: val = ((byte[]) buf)[idx]; val = val < 0 ? val + 256 : val; break;
            case CvType.CV_8S: val = ((byte[]) buf)[idx]; break;
            case CvType.CV_16U: val = ((short[]) buf)[idx]; val = val < 0 ? val + 65536 : val; break;
            case CvType.CV_16S: val = ((short[]) buf)[idx]; break;
            case CvType.CV_32S: val = ((int[]) buf)[idx]; break;
            case CvType.CV_32F: val = ((float[]) buf)[idx]; break;
            default: throw new RuntimeException(String.format("Unimplemented Mat depth %d", depth));
        }
        return val;
    }

    /*
     * Author wyxgoishin
     * Description check Mat in a list whether share same shape and return shape
     * Date 2022/6/15 16:14
     * Param [mats]
     * return int[]
     **/
    private static int[] checkAndGetMatShape(List<Mat> mats){
        int batch = mats.size();
        if(batch == 0){
            return null;
        }
        int[] shape = new int[4];
        for (int b = 0; b < batch; b++) {
            Mat mat = mats.get(b);
            int[] curShape = new int[]{mat.height(), mat.width(), mat.channels(), mat.depth()};
            if(b == 0){
                System.arraycopy(curShape, 0, shape, 0, shape.length);
            }else if(!Arrays.equals(shape, curShape)){
                throw new RuntimeException(String.format("Conflicting Mat shape (height, width, channels, depth): " +
                        "expected %s, got %s at %d-th Mat", Arrays.toString(shape), Arrays.toString(curShape), b));
            }
        }
        return shape;
    }

    /*
     * @Author wyxgoishin
     * @Description Creat a factor array for indexing FloatBuffer
     * @Date 2022/6/13 16:35
     * @Param
     * @Return
     **/
    private static int[] getFactors(int dim1, int dim2, int dim3){
        int[] factors = new int[3];
        factors[2] = dim3;
        factors[1] = dim3 * dim2;
        factors[0] = dim3 * dim2 * dim1;
        return factors;
    }
}
