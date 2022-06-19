package util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Objects;

public class MiscUtil {
    private static boolean isOpencvInited = false;
    private static final String opencvDllName = "opencv_java455.dll";
    private static final String Jar = "jar";
    private static final String FILE = "file";
    private static File fileChecker = null;

    public static void initOpenCV(){
        if(!isOpencvInited){
            try{
                String protocol = Objects.requireNonNull(MiscUtil.class.getResource("")).getProtocol();
                if(Jar.equals(protocol)){
                    loadJarDll(MiscUtil.opencvDllName);
                }else if(FILE.equals(protocol)){
                    System.load(MiscUtil.opencvDllName);
                }
            }catch (Exception e){
                throw new RuntimeException();
            }
            isOpencvInited = true;
        }
    }

    public static void loadJarDll(String name) throws IOException {
        URL url = ClassLoader.getSystemResource(name);
        InputStream in = url.openStream();
        byte[] buffer = new byte[1024];
        File temp = File.createTempFile(System.getProperty("java.io.tmpdir"), name);
        FileOutputStream fos = new FileOutputStream(temp);

        int read;
        while((read = in.read(buffer)) != -1) {
            fos.write(buffer, 0, read);
        }
        fos.close();
        in.close();

        System.load(temp.getAbsolutePath());
    }

    public static boolean exists(String filepath){
        fileChecker = new File(filepath);
        return fileChecker.exists();
    }
}
