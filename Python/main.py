import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Basic MLP with 2 inputs, 4 hidden layers
        # and 10 outputs where each output is
        # the softmax probabilities of a number 0 to 9
        self.MLP = nn.Sequential(
            nn.Linear(2, 5),
            nn.Linear(5, 10),
            nn.Linear(10, 15),
            nn.Linear(15, 20),
            nn.Linear(20, 15),
            nn.Linear(15, 10),
            nn.Softmax(-1)
        )
    
    def forward(self, X):
        return torch.argmax(self.MLP(X), dim=-1)



def main():
    # Create the model
    model = Model()
    
    # Create 4 random noise vectors
    # meaning we want 4 random numbers
    X = torch.distributions.uniform.Uniform(-10000,\
        10000).sample((4, 2))
    
    # Send the noise vectors through the model
    # to get the argmax outputs
    outputs = model(X)
    
    # Print the outputs
    for o in outputs:
        print(f"{o.item()} ")
    
    # Save the model to a file named model.pkl
    torch.save(model.state_dict(), "model.pkl")



def optimizeSave():
    # Load in the model
    model = Model()
    model.load_state_dict(torch.load("model.pkl", \
        map_location=torch.device("cpu")))
    model.eval() # Put the model in inference mode
    
    # Generate some random noise
    X = torch.distributions.uniform.Uniform(-10000, \
        10000).sample((4, 2))
    
    # Generate the optimized model
    traced_script_module = torch.jit.trace(model, X)
    traced_script_module_optimized = optimize_for_mobile(\
        traced_script_module)
    
    # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter(\
        "model.pt")



if __name__ == '__main__':
    main()
    optimizeSave()