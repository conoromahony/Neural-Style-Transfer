# Neural Style Transfer Explorations

by Conor O’Mahony

for CSCI S-89 Final Project, Summer 2024

## Abstract
In this report, I explore the use of Neural Style Transfer (Gatys et al, 2015) to replace the manual steps I undertake when preparing to paint pop-art style portraits. Neural Style Transfer takes the content from a base image, takes the style from a style image, and generates a new image of the chosen content in the chosen style. I implement the Neural Style Transfer algorithm (adapting code from Francois Chollet), which does not produce acceptable results. I then experiment with several aspects of this implementation:
- The choice of optimizer
- The loss function weights
- The choice of layers for measuring content and style

Each of these experiments helped to successively improve the generated image. The most surprising improvements came when experimenting with the choice of layers for measuring the style. On a whim, I experimented with layer combinations I had not seen used in other published materials, and the results were impressive.

Finally, I applied a technique called Spatial Control (Gatys et al, 2016) to further improve the generated image by selectively applying the style to certain regions of the content. The end results met my expectations, and I was able to produce a generated image that I can use to replace the manual work I undertake before putting paint on canvas (converting the image to grayscale, identifying tonal regions, making those regions into geometric shapes, choosing complementary colors for bordering tonal areas).

YouTube Video: https://www.youtube.com/watch?v=FIDVFDB5_Xk

## 1.0 Background
One of my hobbies is painting portraits on canvas with acrylic paint. I like to use a very specific pop-art style with straight lines and contrasting bold colors. To the right, you can see a photograph of me in front of a couple of my paintings.

My goal with this project is to take a photograph of a person as input, and to output an image showing that person in the style of these paintings.

Before we get started, let’s define some terminology:
- Base Image: The image to which we want to apply the style.
- Style Image: The image whose style we want to apply to another image.
- Generated Image: The resulting image, which has the content from Base Image and the style from Style Image.

### 1.1 Selection Criteria
As we go through this paper, we will be choosing among generated images. Admittedly, the selection criteria for choosing desired generated images is subjective. When evaluating images, I use the following criteria:
- The segmentation of the face into distinct areas (based on tone).
- The sharpness of the lines between each segment.
- The choice of colors to represent the different segments.

## 2.0 Introduction to Neural Style Transfer
In 2015, Leon Gatys et al introduced Neural Style Transfer in a paper titled A Neural Algorithm of Artistic Style. The idea behind Neural Style Transfer is to take the “style” of one image and apply it to the “content” of another image. To be able to apply style from one image to another, we need:
- A way to separately identify the content of an image and the style of an image
- A way to measure the content of an image, and therefore minimize the content loss (when compared to the base image)
- A way to measure the style of an image, and therefore minimize the style loss (when compared to the style image)

### 2.1 Identifying the Content and Style of an Image
In their paper, Gatys et al remind us that the higher layers in a convolutional neural network capture the high-level content of the image. Those higher layers capture abstract representations of the objects in the image, as well as the arrangement of those objects. These higher layers don’t constrain the objects to specific pixel locations, and are therefore ideal for capturing the essence of the content. To choose a convolutional layer for comparing the content, Gatys et al recommend choosing one of the higher layers in the network (to best capture the content and arrangement).

Gatys et al also noted that by combining the feature correlations of multiple layers, we can capture the style of an image. The style is defined as the textures, colors, and visual patterns in the image. By using feature correlations that span multiple layers, we can get a representation of the style at multiple scales. Therefore, Gatys et al use the feature correlations across multiple layers to capture the style of an image.

### 2.2 Using a Pre-Trained Model
Gatys et al take advantage of transfer learning, using a pre-trained convolutional neural network that has already been trained to recognize objects in images. In their paper, Gatys et al use the 19-layer VGG-Network (which is referred to as VGG19 in Keras). In particular, the paper uses the 16 convolutional and 5 pooling layers of VGG19. They do not use any of the fully-connected layers.

In particular, here is what they did:
- Set up the model using the VGG19 network with the ImageNet weights.
- Replace the max-pooling layers with average-pooling layers to improve the gradient flow, and produce more appealing results.
- Compute the layer activations for the style image, the base image, and the generated image.
- Define a loss function that captures 1) the content loss when compared to the base image and 2) the style loss when compared to the style image.
- Use gradient descent to minimize that loss function.

### 2.3 Measuring the Content Loss
Gatys et al use a single layer when measuring the content loss. After some experimentation, they found that the conv4_2 layer works best. To measure the content loss, Getys et al use L2 normalization between:
- The conv4_2 layer activations for the base image
- The conv4_2 layer activations for the generated image

By minimizing this content loss, they ensure that the generated image will have similar content to the base image.

### 2.4 Measuring the Style Loss
The style is defined as the textures, colors, and visual patterns in the image. To capture the style information, Gatys et al combine the feature correlations of multiple layers. Because the feature correlations span multiple layers, we get a multi-scale representation of the style. In particular, Gatys et al chose the conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 layers.

### 2.5 Combining the Content and Style Loss
To successfully apply the style of a style image to the content of a base image, we need to minimize the style loss when compared with the style image and minimize the content loss when compared with the base image. Therefore, the loss function must combine the minimization of style loss with the minimization of content loss.

## 3.0 Experiments with Neural Style Transfer
In the file named **1_Neural Style Transfer - Baseline.ipynb**, I adapted code by Francois Chollet (from his book Deep Learning with Python) that implements Neural Style Transfer. This Jupyter Notebook uses the settings from Chollet’s code (that is, it uses the optimizer choice, the loss weight choices, the layer choices, and so on). It takes a base image of Geoffrey Hinton, a style image that uses the desired pop-art style, and applies the neural style transfer algorithm to produce the generated image. Figure 1 provides a summary of the results.

In the next few sections of this report, I will see if I can improve on this baseline generated image. 

### 3.1 Experimenting with the Optimizer
In the implementation of neural style transfer we used above, Chollet uses a Stochastic Gradient Descent optimizer. However, the original Gatys et al neural style transfer algorithm uses Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimization. Because TensorFlow doesn't include support for L-BFGS optimization, Chollet sought alternatives.

In the file named **2_Neural Style Transfer - Optimizer Experiments.ipynb**, I experimented with different optimizer settings that are available in TensorFlow. Figure 2 shows the images generated with the following optimizer settings:
- Stochastic Gradient Descent, with a learning rate of 100 (from Chollet)
- Stochastic Gradient Descent, with a learning rate of 10
- Adam, with a learning rate of 0.001
- Adam, with a learning rate of 0.01
- Adam, with a learning rate of 0.1 and an exponential decay schedule
- Adam, with a learning rate of 0.75 and an exponential decay schedule
- RMSProp, with an initial learning rate of 0.01 and an exponential decay schedule
- RMSProp, with an initial learning rate of 0.1 and an exponential decay schedule
- RMSProp, with an initial learning rate of 0.5 and an exponential decay schedule
- Nadam, with a learning rate of 0.01 and an exponential decay schedule
- Nadam, with a learning rate of 0.1 and an exponential decay schedule
- Nadam, with a learning rate of 0.5 and an exponential decay schedule

Here is my evaluation of the generated images, which is admittedly subjective:
- SGD lr=100 is our baseline image from Section 3.0
- SGD lr=10 is not as good as the baseline (not enough style applied)
- Adam lr=0.001 is not as good as the baseline (not enough style applied)
- Adam lr=0.01 is not as good as the baseline (not enough style applied)
- Adam lr=0.1 is not as good as the baseline (not enough style applied)
- Adam lr=0.5 is not as good as the baseline (not enough style applied)
- RMSProp lr=0.01 is not as good as the baseline (not enough style applied)
- RMSProp lr=0.1 is best
- RMSProp lr=0.5 is better than the baseline
- Nadam lr=0.01 is not as good as the baseline (not enough style applied)
- Nadam lr=0.1 is not as good as the baseline (not enough style applied)
- Nadam lr=0.5 is second best

As we continue our experiments, we will use the RMSProp optimizer with a learning rate of 0.1 and an exponential decay schedule.

### 3.2 Experimenting with the Loss Function
The loss function attempts to minimize both the content loss and the style loss. Let’s experiment with different formulations of the loss function. In particular, let’s experiment with different values for the following variables:
- content_weight
- style_weight
- total_variational_weight

We will see if various combinations of these loss function weights will help improve the generated image. In the file named **3_Neural Style Transfer - Loss Experiments.ipynb**, you can see the code that tests the various combinations of these values. The resulting generated images are available in the Experiments_Loss directory.

At first glance, when examining these images, we can see that:
- In general, the smallest style weight value we experimented with (1e-10) struggles to get the generated images to adopt the desired style.
- In general, for the largest content weight value we experimented with (2.5e-2), the content loss overwhelms the style loss.
- In general, when using the largest total variational weight value (1e-2), the generated images struggle with clarity.

Upon closer examination, it becomes apparent that the best weight combinations are:
- Best: content_weight=2.5e-8, style_weight=1e-2, total_variational_weight=1e-2
- Second best: content_weight=2.5e-10, style_weight=1e-2, total_variational=1e-2
- Third best: content_weight=2.5e-8, style_weight=1e-6, total_variational=1e-6

These weight combinations offer the cleanest delineation between colors for the style we want to apply. As we continue the experiments, we will now switch to using a content_weight of 2.5e-8, a style_weight of 1e-2, and a total_variational_weight of 1e-2.

### 3.3 Experimenting with the Layers
A single layer is used when measuring the content loss. To best capture the content, this will typically be one of the higher convolutional layers in the network. For their paper, Gatys et al chose the conv4_2 layer. In his book, Chollet chose the block5_conv2 layer. We will experiment with different possible content layer choices.

To capture the style information, we combine the feature correlations of multiple layers. For their paper, Gatys et al chose the conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 layers.  In his book, Chollet chose the block1_conv1, block2_conv1, block3_conv1, block4_conv1, and block5_conv1 layers. 

Here are the values with which we will experiment:
- Content layer:
  - Option 1: block5_conv1
  - Option 2: block5_conv2
  - Option 3: block5_conv3
  - Option 4: block5_conv4
- Style layers:
  - Option 1: "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", and "block5_conv1"
  - Option 2: "block1_conv2", "block2_conv2", "block3_conv2", "block4_conv2", and "block5_conv2"

In the file named **4_Neural Style Transfer - Layers Experiments.ipynb**, you can see the experiments. The resulting generated images are available in the Experiments_Layers directory. After trying the above combinations, I added one more, where I chose style layers from different blocks to see what would happen.

From looking at these results, we can see that:
- There is no perceptual difference between the different content layers. We appear to get very similar results regardless of which content layer we test.
- There is a significant difference between style option 1 and style option 2, not only with color choice, but also with the segmentation of the facial structure.
- The additional experiment of mixing different blocks for the style layers is interesting. While the image is more blurred, the color choices and segmentation of the facial structure are different enough to warrant further experimentation (in case it yields a better option). This is an unexpected surprise.

For the content layer, let’s stick with Chollet’s choice of the block5_conv2 layer. But for the style layers, let’s run an additional experiment. Let’s mix-and-match blocks across the layers and see what happens. 

From looking at these results, we can see that:
- It is block1_conv2 that is key to the generated images that have the faces containing more yellow coloring. Admittedly, this is subjective, but the generated images that use block1_conv1 look better. Let’s focus on block1_conv1.
- The generated images that use block2_conv2 clearly have cleaner lines and better segment definitions than those generated using block2_conv1. Let’s keep this additional definition, and choose block2_conv2.
- The blocks for conv3 are interesting. Unlike the other layers, there is not one clear aspect of the style that is discernible. However, the images for each block choice are quite different. The image with the cleanest segments and best color choices is block3_conv3.
- The generated images that use block4_conv2 offer more detail in the chin and cheek areas of the face. Let’s keep this additional definition, and choose block4_conv2.
- It’s very minor, but block5_conv3 and block5_conv4 offer more clarity than block5_conv2. Let’s arbitrarily choose block5_conv3.

I did not anticipate this outcome. Because both Gatys et al and Chollet both chose the same block from each layer, I assumed that doing so was best practice. However, when on a whim I chose to experiment with a generated image from “mixed” blocks, it unexpectedly produced interesting results. This led me to look deeper, and I discovered that “mixing the blocks” from these style layers is an important consideration in achieving our desired style in the generated image. For the record, our chosen set of layers for calculating the style loss is:
- block1_conv1
- block2_conv2
- block3_conv3
- block4_conv2
- block5_conv3

### 3.4 Wrapping Up
This completes our initial attempts to optimize the Neural Style Transfer process. 

## 4.0 Controlling Perceptual Factors
In 2016, Leon Gatys et al improved on the Neural Style Transfer technique in a paper titled Controlling Perceptual Factors in Neural Style Transfer. In particular, they introduced methods for better controlling:
- The style in different parts of the image, which they call spatial control.
- The color choice in the generated image, which they call color control.
- The scale of the applied style, which they call scale control.

### 4.1 Spatial Control
Spatial control involves:
- Using a mask to divide the base image into different regions.
- Using masks to divide one or more style images into different regions.
- Applying different style regions to different regions of the base image. 

The often cited application for spatial control is to improve generated images of landscapes, by applying one style region to the land and a different style region to the sky. Various combinations are possible. When applying style to a base image region, you can use:
- A style region from the same style image as used on another region of base image.
- A style region from a different style image as used on other regions of the base image.
- No style region (to preserve that region from the base image).

Clearly, there are many possible approaches to spatial control. You can apply a style to a portion of an image, you can apply different styles to different portions of an image, and so on. We will implement one of the simpler approaches. We will apply the style to the actual portrait of Geoffrey Hinton, and we will not apply any style to the background.

To implement spatial control, we need to update both our code for training the network, and our code for computing the losses. In the code for training the network, we need to compute the loss and gradients with respect to the masked content, so the style transfer does not affect the masked areas. To ensure that the masked areas remain unchanged, we need to add a term that penalizes any changes to the masked area in the code for computing the losses.

In the file named **Neural Style Transfer - Spatial Experiments.ipynb**, you can see the code for spatial control. Clearly, the image with spatial control looks better.

### 4.2 Color Control and Scale Control
The primary use case for Color Control is to keep the original base image colors, rather than adopting the style image colors. There are many use cases for this when dealing with landscapes, for instance. Obviously, we want to use the colors from the style image, so Color Control does not interest us for this project.

Scale Control allows someone to mix different styles at different scales in a generated image. Again, this is not something we want to do for this project.

## 5.0 Future Work
One obvious place to experiment with, is to consider a different pre-trained convolutional neural network. The VGG-19 network has been trained on images covering a wide variety of subjects (using weights from ImageNet). Perhaps we might get better results from a pre-trained network that uses only images of faces. After a quick search, I did find such a network: VGGFace2, which is trained using 3.3 million images of faces.
Selim et al built upon the work of Gatys et al, fixing some of the structural and color issues found with applying Neural Style Transfer to portraits (rather than general images). They claim their “approach transfers the painting style while maintaining the input photograph identity. In addition it significantly reduces facial deformations over state of the art.” After reading their paper, I don’t believe their approach will offer an improvement over what we have achieved in this paper (for the images we are working with). Their paper fixes artifacts that we have already eliminated from our images. However, it may be worthwhile to implement the Selim et al approach, and experiment with it to verify this is the case.
In 2016, Ulyanov et al and Johnson et al introduced an approach called Real-Time Style Transfer where you build an “image transformation network” that applies a single style. In other words, you build a separate image transformation network for each style you want to apply. While it is cumbersome to build a new image transformation network for each style, Real-Time Style Transfer offers a very important advantage over Neural Style Transfer: it applies styles to base images quickly. Neural Style Transfer can be quite slow, taking approximately an hour to stylize a single base image. If you pre-train a Real-Time Style Transfer image transformation network, applying that style to a base image is relatively fast. Instead of repeatedly passing the images forward and backward through the network, Real-Time Style Transfer generates the stylized image after a single forward pass through the network. Implementing Real-Time Style Transfer for our desired style will allow us to quickly apply that style to other portraits.

## 6.0 Files for Running Code
The code for this report is in the following files:
- **1_Neural Style Transfer - Baseline.ipynb** is a Jupyter Notebook with code that applies neural style transfer to a base image. The code was adapted from Francois Chollet’s book Deep Learning with Python. The code in this file corresponds to Section 3.0 of the report.
- **2_Neural Style Transfer - Optimizer Experiments.ipynb** is a Jupyter Notebook with code that experiments with different optimizers, and different optimizer settings. The code in this file corresponds to Section 3.1 of the report.
- **3_Neural Style Transfer - Loss Experiments.ipynb** is a Jupyter Notebook with code that experiments with different loss function weight settings. The code in this file corresponds to Section 3.2 of the report.
- **4_Neural Style Transfer - Layers Experiments.ipynb** is a Jupyter Notebook with code that experiments with different layer choices. The code in this file corresponds to Section 3.3 of the report.
- **5_Neural Style Transfer - Optimized.ipynb** is a Jupyter Notebook with code that takes the learnings from each of the experiments and prepares an optimized generated image. The code in this file corresponds to Section 3.4 of the report.
- **6_Perceptual Control.ipynb** is a Jupyter Notebook with code that applies Spatial Control (from the second Gatys et al paper) to improve the appearance of the portrait background. The code in this file corresponds to Section 4.1 of the report.

This code uses the following directories:
- The Images directory contains the following images:
  - **Hinton.jpg**, which is our base image.
  - **GeorgeFloyd.jpg**, which is our style image.
  - **Baseline Generated Image.png**, which is our initial generated image.
  - **Optimized Generated Image.png**, which is the generated image after optimizing hyperparameters.
  - **Hinton_mask.jpg**, which is a mask we use for spatial control.
  - **Spatial Control Image.png**, which is the result of applying spatial control to our optimized image.
- The **Experiments_Optimizer** directory contains images that were generated by the code in **2_Neural Style Transfer - Optimizer Experiments.ipynb**.
- The **Experiments_Loss** directory contains images that were generated by the code in **3_Neural Style Transfer - Loss Experiments.ipynb**.
- The **Experiments_Layers** directory contains images that were generated by the code in **4_Neural Style Transfer - Layers Experiments.ipynb**.

## 7.0 References
Dmitry Ulyanov, Vadim Lebedev, Andrea Vedaldi, Victor Lempitsky. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images. arXiv (2016). https://arxiv.org/pdf/1603.03417

Francois Chollet. Deep Learning with Python, Second Edition. Pages 383-391. Manning Press (2021).

Justin Johnson, Alexandre Alahi, Li Fei-Fei. Perceptual Losses for Real-time Style Transfer and Super-resolution. arXiv (2016). https://arxiv.org/pdf/1603.08155

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. A Neural Algorithm of Artistic Style. arXiv (2015). https://arxiv.org/pdf/1508.06576

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann, Eli Shechtman. Controlling Perceptual Factors in Neural Style Transfer. arXiv (2016). https://arxiv.org/pdf/1611.07865

Ahmed Selim, Mohamed Elgharib, Linda Doyle. Painting Style Transfer for Head Portraits using Convolutional Neural Networks. ACMTrans. Graph 35, 4, Article 129 (July 2016). https://dl.acm.org/doi/10.1145/2897824.2925968

Thiago Ambiel. Portrait Stylization: Artistic Style Transfer with Auxiliary Networks for Human Face Stylization. arXiv (2023). https://arxiv.org/pdf/2309.13492

Yongcheng Jing, Yezhou Yang, Zunlei Feng, Jingwen Ye, Yizhou Yu, Mingli Song. Neural Style Transfer: A Review. arXiv (2018). https://arxiv.org/pdf/1705.04058
