import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OnnxTensor;

import org.opencv.core.Mat;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_UNCHANGED;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Map;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;
import java.io.IOException;

import static util.MiscUtil.exists;
import static util.MiscUtil.initOpenCV;
import static util.OnnxTensorUtil.createTensorFromImageMat;
import static util.OnnxTensorUtil.flowTensorToKittiMat;
import static util.OnnxTensorUtil.defaultTensorDimOrders;

public class SampleRAFT {
    private static final Logger logger = LoggerFactory.getLogger(SampleRAFT.class);
    private static final String HELP = "help";
    private static final String INFERENCE = "inference";
    private static final String LOAD = "load";
    private static final String ONNX = "onnx";
    private static final String PNG = "png";
    private static final String QUIT = "quit";
    private static final String USAGE = "Usage:\n  " +
                                            "help\n  " +
                                            "inference <path-to-image1> <path-to-image2> <path-to-save>\n  " +
                                            "load <path-to-model> [cuda-device-num]\n  " +
                                            "quit\n";
    private OrtEnvironment env;
    private SessionOptions opts;
    private OrtSession session;
    private boolean modelLoaded;

    public SampleRAFT() throws OrtException {
        initOpenCV();
        this.env = OrtEnvironment.getEnvironment();
        this.opts = new SessionOptions();
        this.opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
    }

    public void setCuda(int deviceNum) throws OrtException {
        this.opts.addCUDA(deviceNum);
        logger.info("Add gpu device {} for inference", deviceNum);
    }

    public void loadModel(String modelPath) throws OrtException {
        if(this.session != null){
            this.session.close();
        }

        logger.info("Loading model from {}", modelPath);
        this.session = env.createSession(modelPath, this.opts);

        logger.info("Inputs:");
        for (NodeInfo i : session.getInputInfo().values()) {
            logger.info(i.toString());
        }

        logger.info("Outputs:");
        for (NodeInfo i : session.getOutputInfo().values()) {
            logger.info(i.toString());
        }

        this.modelLoaded = true;
    }

    public void inference(String imgPath1, String imgPath2, String savePath) throws OrtException {
        if(!this.modelLoaded){
            logger.warn("Try to do inference before loading model, skipping this operation");
            return;
        }

        if(!savePath.endsWith(PNG)){
            logger.warn("As flow will be saved in kitti-format, only '.png' is supported. Skip this operation.");
            return;
        }

        // create parent directory of save-path in advance
        File file = new File(savePath);
        Deque<File> stack = new ArrayDeque<>();
        while((file = file.getParentFile()) != null && !file.exists()){
            stack.push(file.getParentFile());
        }
        while(!stack.isEmpty()){
            File dir = stack.pop();
            if(dir.mkdir()){
                logger.info("Create parent directory of save-path: {}", file.getAbsolutePath());
            }else{
                logger.error("Unable to create parent directory of save-path: {}, skipping this operation", file.getAbsolutePath());
                stack.clear();
                return;
            }
        }

        Mat mat1 = imread(imgPath1, IMREAD_UNCHANGED);
        Mat mat2 = imread(imgPath2, IMREAD_UNCHANGED);

        if(mat1.height() != mat2.height() || mat1.width() != mat2.width() || mat1.channels() != mat2.channels()){
            logger.error("Conflicting input image shape of {} and {}", mat1.size(), mat2.size());
            mat1.release();
            mat2.release();
            return;
        }

        OnnxTensor tensor1 = createTensorFromImageMat(this.env, mat1, defaultTensorDimOrders);
        OnnxTensor tensor2 = createTensorFromImageMat(this.env, mat2, defaultTensorDimOrders);
        Map<String, OnnxTensor> inputs = Map.of("image1", tensor1, "image2", tensor2);

        Result result = session.run(inputs);
        OnnxTensor output = (OnnxTensor) result.get(0);

        Mat matRet = flowTensorToKittiMat(output, defaultTensorDimOrders);
        if(imwrite(savePath, matRet)){
            logger.info("Save kitti-format flow-prediction to: {}", savePath);
        }else{
            logger.warn("Failed to save kitti-format flow-prediction to: {}. Check whether the extension " +
                    "of save-path is right.", savePath);
        }

        matRet.release();
        output.close();
        // result.close(); // doing this will close the VM

        mat1.release();
        mat2.release();
        tensor1.close();
        tensor2.close();
    }

    public void close() throws OrtException {
        if(this.session != null){
            this.session.close();
        }
        this.opts.close();
        this.env.close();
    }

    public static void main(String[] args) throws OrtException {
        System.out.print(USAGE);
        SampleRAFT raft = new SampleRAFT();
        while (true){
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
            try {
                String line = bufferedReader.readLine();
                String[] operation = line.split(" ");
                String opCode = operation[0];
                if(HELP.equals(opCode)){
                    System.out.print(USAGE);
                }else if(INFERENCE.equals(opCode)){
                    if(operation.length < 4){
                        logger.warn("Expected 4 argument for inference operation, got {} instead.", operation.length);
                    }else{
                        boolean allFileExists = true;
                        for(int i = 1; i <= 2 && allFileExists; i++){
                            if(!exists(operation[i])){
                                logger.error("Given <path-to-image{}> '{}' does not exists, skipping this operation", i, operation[i]);
                                allFileExists = false;
                            }
                        }
                        if(allFileExists){
                            raft.inference(operation[1], operation[2], operation[3]);
                        }
                    }
                }else if(LOAD.equals(opCode)){
                    if(operation.length > 2) {
                        int deviceNum = Integer.parseInt(operation[2]);
                        if (deviceNum < 0) {
                            logger.warn("Expected cuda device num to be none negative integer, got {} instead. Will skip cuda setting.", deviceNum);
                        } else {
                            raft.setCuda(deviceNum);
                        }
                    }

                    String modelPath = operation[1];
                    if(!modelPath.endsWith(ONNX)){
                        logger.error("Only onnx models are supported currently.");
                        continue;
                    }

                    if(!exists(modelPath)){
                        logger.error("Given model path '{}' not exists, skipping this operation.", modelPath);
                    }
                    raft.loadModel(operation[1]);
                }else if(QUIT.equals(opCode)){
                    raft.close();
                    break;
                }else{
                    logger.warn("Unknown operation: {}", line);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

}
