import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

from MyNet import my_ConvNet

# Load the model
net = my_ConvNet()
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
net.eval()

# Load the image
img = cv2.imread('./test_pics/0.png', 0)

# Pre-process the image
img = cv2.resize(img, (28,28))
img = 255 - img
T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)) ])
img = T(img)
img = img.unsqueeze(0)

# Use your image as input to run the model
with torch.no_grad():
    predictions = net(img)

# Post-process the predictions
predictions = predictions.cpu().numpy()
outcome = np.argmax(predictions)

# Print out the final output
print("it's number {}".format(outcome))