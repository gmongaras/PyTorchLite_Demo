from torch.utils.mobile_optimizer import optimize_for_mobile
from random import randint
import torch
from torch import nn
import dnnlib
import numpy as np
import legacy
import re



# Wrapper around the stylegan model
class wrapper(nn.Module):
    def __init__(self):
        super(wrapper, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load in a given model
    def load(self, networkPklPath):
        with dnnlib.util.open_url(networkPklPath) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
    
    
    
    # Generate image given some noise
    def forward(self, z):
        label = torch.zeros([1, self.G.c_dim], device=self.device)

        img = self.G(z, label, truncation_psi=1, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128)#.clamp(0, 255).to(torch.int8)
        return img[0]
    
    
    def make_transform(translate, angle):
        m = np.eye(3)
        s = np.sin(angle/360.0*np.pi*2)
        c = np.cos(angle/360.0*np.pi*2)
        m[0][0] = c
        m[0][1] = s
        m[0][2] = translate[0]
        m[1][0] = -s
        m[1][1] = c
        m[1][2] = translate[1]
        return m
    
    
    def parse_range(s):
        '''Parse a comma separated list of numbers or ranges and return a list of ints.

        Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
        '''
        if isinstance(s, list): return s
        ranges = []
        range_re = re.compile(r'^(\d+)-(\d+)$')
        for p in s.split(','):
            m = range_re.match(p)
            if m:
                ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
            else:
                ranges.append(int(p))
        return ranges
        


def main():
    # The location of the model to load
    networkPklPath = "./training_runs/run/network-snapshot-000000.pkl"



    # The wrapper for the stylegan model
    model = wrapper()


    # Load in the model
    model.load(networkPklPath)


    # Generate the image as a 3-D tensor
    #t = generate_images(["--network", networkPklPath, "--seeds", seed, "--outdir", "./output"])
    # t = model(torch.ones(1, 512)).to(torch.uint8)
    # import PIL
    # from PIL import Image
    # PIL.Image.fromarray(t.cpu().numpy(), 'RGB').save(f'output/seed.png')
    
    # Generate the optimized model
    seed = torch.tensor(1)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, model.G.z_dim)).to(model.device)
    traced_script_module = torch.jit.trace(model, z)
    traced_script_module_optimized = optimize_for_mobile(\
        traced_script_module)
    
    # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter(\
        "imageGen.pt")




if __name__ == "__main__":
    main()