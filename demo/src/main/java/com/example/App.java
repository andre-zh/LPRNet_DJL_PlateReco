package com.example;

import com.example.utils.translator.*;
import java.nio.file.*;
import java.io.IOException;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.translate.*;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException
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
