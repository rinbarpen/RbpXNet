import logging
import os
import os.path

import numpy as np
import torch
from PIL import Image

from utils.utils import create_dirs, load_model
from utils.visualization import draw_attention_heat_graph, draw_heat_graph


class Predictor:
    def __init__(self, net, classes, device=None):
        from config import CONFIG
        
        self.device = device if device else CONFIG['device']
        self.net = net.to(self.device)
        self.classes = classes
        
        assert len(self.classes) >= 1, 'The number of classes should be greater than 0'

    @torch.inference_mode()
    def predict_one(self, input):
        from config import CONFIG

        self.net.load_state_dict(load_model(CONFIG['load'], device=self.device)['model'])

        # input = Image.open(input).convert('RGB')
        # original_size = input.size # (W, H)
        # input = input.resize((512, 512))  # TODO: to be more flexible
        # input = torch.from_numpy(np.array(input).transpose(2, 0, 1))
        # input = input.unsqueeze(0) # (1, 3, 512, 512)

        input = Image.open(input).convert('L')
        original_size = input.size
        input = input.resize((512, 512))
        input = torch.from_numpy(np.array(input))
        input = input.unsqueeze(0).unsqueeze(0) # (1, 1, 512, 512)
        
        input = input.to(self.device, dtype=torch.float32)

        self.net.eval()
        
        predict = self.net(input)  # (1, N, H, W)
        # predict = F.sigmoid(predict)
        # predict = (predict - predict.min()) / (predict.max() - predict.min())
        # predict = predict.squeeze(0)
        predict = predict.squeeze(0)
        possibilities = predict.cpu().detach().numpy()  # (N, H, W)
        predict_image = possibilities.copy()  # (N, H, W)

        predict_image[predict_image >= 0.5] = 255
        predict_image[predict_image < 0.5] = 0
        predict_image = predict_image.astype(np.uint8)

        assert predict_image.shape[0] == len(self.classes), "The number of classes should be equal to the number of ones predicted"
        result = dict()
        for i, name in enumerate(self.classes):
            possibility = possibilities[i]
            mask = Image.fromarray(predict_image[i], mode='L').resize(size=original_size)
            result[name] = {
                "possibility": possibility,
                "mask": mask,
            }
        return result


    def predict_more(self, inputs):
        results = dict()
        for input in inputs:
            result = self.predict_one(input)
            results[input.absolute()] = result

        return results


    def predict(self, inputs):
        from config import CONFIG
        
        for input in inputs:
            result = self.predict_one(input)

            predict_dir = CONFIG["save"]["predict_dir"]
            filename = input.stem
            predict_filename = f"{filename}_predict.png"
            heat_filename = f"{filename}_heat.png"
            fuse_filename = f"{filename}_fuse.png"
            for category, values in result.items():
                category_dir = os.path.join(predict_dir, category)
                predict_path = os.path.join(category_dir, predict_filename)
                heat_path = os.path.join(category_dir, heat_filename)
                fuse_path = os.path.join(category_dir, fuse_filename)
                title = filename

                create_dirs(category_dir)

                possibility = values["possibility"]
                mask_image = values["mask"]
                mask_image.save(predict_path)
                original_image = Image.open(input).convert('RGB')
                logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
                draw_heat_graph(possibility,
                                filename=heat_path,
                                title=title)
                logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
                draw_attention_heat_graph(mask_image,
                                        original_image,
                                        filename=fuse_path,
                                        title=title)
                logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")

        # results = self.predict_more(inputs)
        # for path, result in results:
        #     predict_dir = CONFIG["save"]["predict_dir"]
        #     filename = file_prefix_name(path)
        #     predict_filename = f"{filename}_predict.png"
        #     heat_filename = f"{filename}_heat.png"
        #     fuse_filename = f"{filename}_fuse.png"
        #     for category, values in result.items():
        #         category_dir = os.path.join(predict_dir, category)
        #         predict_path = os.path.join(category_dir, predict_filename)
        #         heat_path = os.path.join(category_dir, heat_filename)
        #         fuse_path = os.path.join(category_dir, fuse_filename)
        #         title = filename

        #         create_dirs(category_dir)

        #         possibility = values["possibility"]
        #         mask_image = values["mask"]
        #         mask_image.save(predict_path)
        #         original_image = Image.open(input).convert('RGB')
        #         logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
        #         draw_heat_graph(possibility,
        #                         filename=heat_path,
        #                         title=title)
        #         logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
        #         draw_attention_heat_graph(mask_image,
        #                                 original_image,
        #                                 filename=fuse_path,
        #                                 title=title)
        #         logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")
