package com.example.utils.translator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.*;
import ai.djl.ndarray.*;
import ai.djl.translate.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.*;

/**
 * A result of all joints found during Human Pose Estimation on a single image.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Articulated_body_pose_estimation">Wikipedia</a>
 */
public class PlateTranslator implements Translator<Image, String> {

    private char[] dict = { '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                            '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                            '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                            '新',
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                            'W', 'X', 'Y', 'Z', 'I', 'O', '-'};

    public PlateTranslator() {

    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray imageND = input.toNDArray(ctx.getNDManager());

        // preprocess
        imageND = NDImageUtils.resize(imageND, 94, 24)
            .sub(127.5f)
            .mul(0.0078125f)
            .transpose(2,0,1);

        // RGB2BGR
        imageND = imageND.get(new NDIndex("2,:,:"))
            .stack(imageND.get(new NDIndex("1,:,:")), 0)
            .concat(imageND.get(new NDIndex("0,:,:")).expandDims(0),0);

        return new NDList(imageND);
    }

    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        NDArray array = list.get(0);

        // postprocess
        String plate = "";
        int n = (int)array.get(new NDIndex(":, 0")).argMax().getAsString().charAt(0);
        int current = n;
        if (current != dict.length - 1){
            plate = plate + dict[current];
        }
    
        for (int i = 0; i < array.getShape().get(1); i++) {
            int c = (int)array.get(new NDIndex(":, "+i)).argMax().getAsString().charAt(0);
            if (c == current || c == dict.length-1 ) {
                if (c == dict.length-1) {
                    current = c;
                }
                continue;
            }
            plate = plate + dict[c];
            current = c;
        }
        return plate;
    }
}