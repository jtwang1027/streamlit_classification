# add models https://pytorch.org/docs/stable/torchvision/models.html







def predict(filename, model_sel):
    '''takes filename + pretrained model string, predicts using  pytorch
    and returns tuple (top category, top probability, all probabilities)'''
    import torch
    import json
    from torchvision import models

    model = torch.hub.load('pytorch/vision:v0.5.0', model_sel, pretrained=True)
    '''
    if model_sel=='resnet18':
        model= models.resnet18(pretrained=True)
    elif model_sel=='alexnet':
        model= models.alexnet(pretrained=True)
    elif model_sel=='vgg16': #not vgg11
        model= models.vgg16(pretrained=True)
    else:
        # model_sel=='mobilenet_v2':p
        model= models.mobilenet_v2(pretrained=True)'''


    model.eval()

    file_read = open("imagenet_class_index.json").read()
    categ = json.loads(file_read)
    #print(categ['0'])

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    

    '''
    if len(imread(filename))==2:
        #if image is grayscale, convert to rgb
        input_image=input_image.convert('RGB')
    '''

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    with torch.no_grad(): #from torch package
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probab=torch.nn.functional.softmax(output[0], dim=0)
    idx=sorted(range(len(probab)), key=lambda i: probab[i])[-1] #top predictions
    
    most_prob=str(probab[idx].numpy()) #probability of most likely category
    print(categ[str(idx)])
    print(most_prob)
    print('finished')
    return( (categ[str(idx)][1], most_prob, probab))

#testing all models
model_list=['alexnet', 'resnet18',  'vgg16',  'mobilenet_v2' ]

'''for mo in model_list:
    print(mo)
    x,y, _ = predict('/home/jason/590-Final-Project/docker-classification/sample_img/val_3.JPEG',model_sel=None)

    print((x,y))'''
    
