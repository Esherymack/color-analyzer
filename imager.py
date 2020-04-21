import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def get_image(image_path):
    image = cv2.imread(image_path)
    # By default, OpenCV reads image sequence as BGR
    # To view the actual image we need to convert
    # to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

c = 0
lexicolist = [i for i in range(0, 451)]
lexicolist.sort(key=str)
for filename in os.listdir("./data"):
    if filename.endswith(".jpg"):
        
        fname = str(lexicolist[c]) + ".jpg"
        c += 1
        
        first_pie = get_image(f'./data_runs/first/piecharts/{fname}')
        second_pie = get_image(f'./data_runs/second/piecharts/{fname}')
        third_pie = get_image(f'./data_runs/third/piecharts/{fname}')
        fourth_pie = get_image(f'./data_runs/fourth/piecharts/{fname}')
        fifth_pie = get_image(f'./data_runs/fifth/piecharts/{fname}')
        
        first_peak = get_image(f'./data_runs/first/peaks/{fname}')
        second_peak = get_image(f'./data_runs/second/peaks/{fname}')
        third_peak = get_image(f'./data_runs/third/peaks/{fname}')
        fourth_peak = get_image(f'./data_runs/fourth/peaks/{fname}')
        fifth_peak = get_image(f'./data_runs/fifth/peaks/{fname}')
        
        pie_images = [first_pie, second_pie, third_pie, fourth_pie, fifth_pie]
        peak_images = [first_peak, second_peak, third_peak, fourth_peak, fifth_peak]

        fig, ax = plt.subplots()

        plt.figure(figsize=(20,5), dpi=199)        
        for i in range(len(pie_images)):
            plt.subplot(1, len(pie_images), i+1)
            plt.axis('off')
            plt.grid(b=None)
            plt.imshow(pie_images[i])
        plt.savefig(f"./nbimg/pie-{fname}")
        plt.cla()
        plt.clf()
        plt.close(fig)
        plt.close('all')
              
        plt.figure(figsize=(20,5), dpi=199)
        for i in range(len(peak_images)):
            plt.subplot(1, len(peak_images), i+1)
            plt.axis('off')
            plt.grid(b=None)
            plt.imshow(peak_images[i])

        plt.savefig(f"./nbimg/peak-{fname}")
        plt.cla()
        plt.clf()
        plt.close(fig)
        plt.close('all')
    