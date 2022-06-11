package com.example;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.example.utils.translator.*;
import java.nio.file.*;
import java.io.IOException;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.translate.*;


/**
 * Unit test for simple App.
 */


// import com.example.utils.*;
/**
 * Hello world!
 *
//  */
// public class AppTest 
// {
//     @Test
//     public static void test( )
//     {
//         System.out.println( "Hello World!" );
//         ImageProcessor ip = new ImageProcessor();
//         String s = new String("D:/dev_local/DeepJava/demo/src/main/java/com/example/resources/image/logo.png");
//         ip.smooth(s);
//         /* smooth('D:/dev_local/DeepJava/demo/src/main/resources/image/logo.png');
//         */
//     }
// }

public class AppTest 
{
    /**
     * Rigorous Test :-)
     */
    @Test
    public void test() throws IOException
    {
        try {
            // 加载模型
            Model model = Model.newInstance("LPR");
            Path mp = Paths.get("D:/dev_local/DeepJava/demo/src/main/java/com/example/resources/model/LPR_cpu_torchscript.pt");
            model.load(mp);

            // 定义translator和predictor
            Translator<Image, String> translator = new PlateTranslator();
            Predictor<Image, String> predictor = model.newPredictor(translator);

            // 加载图片
            Image img = ImageFactory.getInstance().fromFile(Paths.get("D:/dev_local/DeepJava/demo/src/main/java/com/example/resources/image/plate_2.png"));
            
            // 获取结果
            String result = predictor.predict(img);

            System.out.println(result);
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
