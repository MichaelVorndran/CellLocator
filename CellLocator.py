import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import multiprocessing
import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel, Menu
import customtkinter as ctk
import tensorflow as tf
import numpy as np
import cv2
import multiprocessing as mp  
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import math
from tqdm import tqdm
import configparser
import webbrowser




config = configparser.ConfigParser() 
config.read('config.ini')

resource_allocation = int(config['DEFAULT']['resource_allocation'])
crop_size = int(config['DEFAULT']['crop_size'])
crop_border = int(config['DEFAULT']['crop_border'])


#pyinstaller --onefile PyInstaller_v5_3.py --distpath E:\PyInstallerTests\v5_3 





class analysed_image_fsc:
    def __init__(self, name, tot_alive_int, avg_alive_int, tot_dead_int, avg_dead_int):
        self.name = name
        self.tot_alive_int = tot_alive_int
        self.avg_alive_int = avg_alive_int
        self.tot_dead_int = tot_dead_int
        self.avg_dead_int = avg_dead_int

class analysed_image:
    def __init__(self, imagename, conf_alive, conf_dead, conf_total, avg_alive_area, avg_dead_area, count_alive, count_dead, count_total, analysed_fsc):
        self.imagename = imagename
        self.conf_alive = conf_alive
        self.conf_dead = conf_dead
        self.conf_total = conf_total
        self.avg_alive_area = avg_alive_area
        self.avg_dead_area = avg_dead_area
        self.count_alive = count_alive
        self.count_dead = count_dead
        self.count_total = count_total
        self.analysed_fsc = analysed_fsc

class flourescence_channel_image:
    def __init__(self, name, org_color, org_gray, normalized_gray, normalized_filterd_alive, normalized_filterd_dead, normalized_filterd_conf, denoised_gray, denoised_filterd_alive, denoised_filterd_dead, denoised_filterd_conf):
        self.name = name
        self.org_color = org_color
        self.org_gray = org_gray
        self.normalized_gray = normalized_gray
        self.normalized_filterd_alive = normalized_filterd_alive
        self.normalized_filterd_dead = normalized_filterd_dead
        self.normalized_filterd_conf = normalized_filterd_conf
        self.denoised_gray = denoised_gray
        self.denoised_filterd_alive = denoised_filterd_alive
        self.denoised_filterd_dead = denoised_filterd_dead
        self.denoised_filterd_conf = denoised_filterd_conf

class flourescence_average_values:
    def __init__(self, org, normalized, denoised):
        self.org = org
        self.normalized = normalized
        self.denoised = denoised

class analysed_cell:
    def __init__(self, x, y, state, avg_flourrescence_intesities):
        self.x = x
        self.y = y
        self.state = state
        self.avg_flourrescence_intesities = avg_flourrescence_intesities 
        
        

global_model = None

def get_timestamp_incuCyte(h):
    days = h // 24  # Calculate whole days
    hours = h % 24  # Calculate remaining hours

    timestamp = f"{days:02d}d{hours:02d}h00m"  # Use f-strings for formatting

    return timestamp



def get_pos_contours(img, erode_kernel_size=3, threshold=10):
    """
    Finds contours in a grayscale image and returns the center positions.

    Args:
        img: The input grayscale image.
        erode_kernel_size: Size of the kernel for erosion (odd number).
        threshold: Threshold value for binary thresholding.

    Returns:
        A list of (x, y) center positions of the detected contours.
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))

    # Direct grayscale conversion (if necessary)
    if len(img.shape) == 3 and img.shape[2] > 1:
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayimg = img.copy()  # Avoid modifying the input if it's already grayscale

    # Erosion
    eroded_img = cv2.erode(grayimg, kernel)

    ## Thresholding
    #ret, thresh = cv2.threshold(eroded_img, threshold, 255, cv2.THRESH_BINARY)  

    # Find contours
    contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pos = []
    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)

        # Calculate x,y coordinate of center (avoiding potential division by zero)
        cX = int(M["m10"] / max(M["m00"], 0.00001)) + 1 
        cY = int(M["m01"] / max(M["m00"], 0.00001)) + 1
        pos.append((cX, cY))

    unique_pos = list(set(pos))
    return unique_pos

def get_cell_count(positions, gray_img_alive, gray_img_dead, size=3):

    alive_count, dead_count, unclear_count = 0, 0, 0
    img_h, img_w = gray_img_alive.shape

    for x, y in positions:
        # Adjust position to stay within bounds
        x = np.clip(x, size, img_w - size - 1)
        y = np.clip(y, size, img_h - size - 1)

        area_alive = gray_img_alive[y-size:y+size, x-size:x+size]
        area_dead = gray_img_dead[y-size:y+size, x-size:x+size]

        sum_alive = np.sum(area_alive)
        sum_dead = np.sum(area_dead)

        if sum_alive > sum_dead:
            alive_count += 1
        elif sum_dead > sum_alive:
            dead_count += 1
        else:
            unclear_count += 1

    return alive_count, dead_count, unclear_count


def get_cell_state(positions, gray_img_alive, gray_img_dead, size=3):

    alive_count, dead_count, unclear_count = 0, 0, 0
    img_h, img_w = gray_img_alive.shape

    cells = []
    for x, y in positions:
        # Adjust position to stay within bounds
        x = np.clip(x, size, img_w - size - 1)
        y = np.clip(y, size, img_h - size - 1)

        area_alive = gray_img_alive[y-size:y+size, x-size:x+size]
        area_dead = gray_img_dead[y-size:y+size, x-size:x+size]

        sum_alive = np.sum(area_alive)
        sum_dead = np.sum(area_dead)

        ac = analysed_cell(x,y,None,None)

        if sum_alive > sum_dead:
            alive_count += 1
            ac.state = 'alive'
            cells.append(ac)

        elif sum_dead > sum_alive:
            dead_count += 1
            ac.state = 'dead'
            cells.append(ac)

        else:
            unclear_count += 1
            ac.state = 'unclear'

        #cells.append(ac)

    return alive_count, dead_count, unclear_count, cells

def gray2rgba_mask(grayimg, output_color):
    ret, mask = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY)   #cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    grayimg_bgr = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2RGB)
    grayimg_bgr[mask == 255] = output_color #[0, 0, 255] = red    [0, 255, 0] = green    [255, 0, 0] = blue
    
    tmp = cv2.cvtColor(grayimg_bgr, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(grayimg_bgr)
    rgba = [b,g,r, alpha]
    grayimg_bgra = cv2.merge(rgba,4)

    return grayimg_bgra

def color2rgba_mask(colorimg, output_color):
    grayimg = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY)   #cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    #grayimg_bgr = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2RGB)
    colorimg[mask == 255] = output_color #[0, 0, 255] = red    [0, 255, 0] = green    [255, 0, 0] = blue
    
    tmp = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(colorimg)
    rgba = [b,g,r, alpha]
    grayimg_bgra = cv2.merge(rgba,4)

    return grayimg_bgra













def init_worker(model_path):
    global global_model
    global_model = tf.lite.Interpreter(model_path=model_path)
    global_model.allocate_tensors()




def create_crops(image, crop_size, crop_border, org_h, org_w, border_mode=cv2.BORDER_REFLECT_101):
    """
    Creates crops for high-quality predictions, replicating your original calculation logic.

    Args:
        image (np.ndarray): The input image.
        crop_size (int): The desired size of the output crops.
        border_width (int): Width of the border region to discard.
        border_mode (int, optional): Border mode for the OpenCV border function.
                                     Defaults to cv2.BORDER_REFLECT_101.

    Returns:
        list: A list of high-quality inner crop images (NumPy arrays).
    """

    # Add mirrored border
    image_with_border = cv2.copyMakeBorder(image, crop_border, 500, crop_border, 500, border_mode)

    inner_crop_size = int(crop_size - 2*crop_border)
    

    # Calculate necessary border for the original image (mirroring your code)
    num_cutouts_w = math.ceil(org_w / inner_crop_size)

    num_cutouts_h = math.ceil(org_h / inner_crop_size)

    crops = []
    w_range = int(num_cutouts_w * inner_crop_size)
    h_range = int(num_cutouts_h * inner_crop_size)

    for i in range(0, w_range, inner_crop_size):
        for j in range(0, h_range, inner_crop_size):
            crop = image_with_border[j:j + crop_size, i:i + crop_size]
            crops.append(crop)


    return crops, num_cutouts_w, num_cutouts_h

def restore_image_from_crops(crops, crop_size, crop_border, num_cutouts_w, num_cutouts_h, org_h, org_w):

    h = int(num_cutouts_h * (crop_size-crop_border))
    w = int(num_cutouts_w * (crop_size-crop_border))
    inner_crop_size = int(crop_size - 2*crop_border)

    background = np.zeros((h,w,3), np.float32) 


    crop_counter = 0
    for i in range(num_cutouts_w):
        for j in range(num_cutouts_h):

            crop = crops[crop_counter]

            inner_crop = crop[crop_border:-crop_border, crop_border:-crop_border, :]

            x1 = 0 + i*inner_crop_size
            x2 = inner_crop_size + i*inner_crop_size
            y1 = 0 + j*inner_crop_size
            y2 = inner_crop_size + j*inner_crop_size

            background[y1:y2, x1:x2, :] = inner_crop

            crop_counter+=1

    return background[0:org_h, 0:org_w, :]




class CellAnalyzer:
    def __init__(self):
        self.mask_dir = str(config['DEFAULT']['mask_dir'])

        self.model_path = str(config['DEFAULT']['model_path'])
        self.modelname = ''

        self.main_dir = ''
        self.phase_img_dir = ''
        self.VID = ''
        self.flourescence_channel_names = config['DEFAULT'].get('flourescence_channel_names').split(',')
        self.flourescence_channel_names_found = []
        self.flourescence_channel_dirs = []



    def on_select_bf_dir_clicked(self):
        self.main_dir = filedialog.askdirectory()
        self.phase_img_dir = os.path.join(self.main_dir,'phase')
        self.flourescence_channel_names_found = []

        filenames = os.listdir(self.phase_img_dir)
        if filenames:  
            self.VID = filenames[0].split('_', 1)[0]
            print(self.VID)

        for fcn in self.flourescence_channel_names:
            full_path = os.path.join(self.main_dir, fcn)
            if os.path.isdir(full_path):
                self.flourescence_channel_dirs.append(full_path)
                self.flourescence_channel_names_found.append(fcn)
                if set(os.listdir(self.phase_img_dir)) != set(os.listdir(full_path)):
                    print(f"The number and/or the names of phase images and the {fcn} flourescence channel don't match! Make sure you want to continue with the analysis.") 



    def on_analyse_clicked(self):
        images = [os.path.join(self.phase_img_dir, img) for img in os.listdir(self.phase_img_dir)]
        total_images = len(images)
        
        pbar = tqdm(total=total_images)
        
        config = configparser.ConfigParser()
        config.read('config.ini')
        resource_allocation = int(config['DEFAULT']['resource_allocation'])
        model_path = str(config['DEFAULT']['model_path'])
        
        
        num_processes = int(max((mp.cpu_count() * (resource_allocation / 100)),1))
        if num_processes > 1:
            print("Starting", num_processes, " processes")
        else:
            print("Starting 1 process")


        csv_dir = os.path.join(self.main_dir, 'cell_data')
        os.makedirs(csv_dir, exist_ok=True) 
        
        results = []
        cell_results_df = []
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
            for result_tuple  in pool.imap_unordered(self.process_single_image, images):
                results_dict, analysed_cells_df = result_tuple
                results.append(results_dict)
                cell_results_df.append(analysed_cells_df)

                pbar.update(1)
        
        pbar.close()
        print("Processing complete.")


        dfac = pd.concat(cell_results_df, ignore_index=True)
        #pd.set_option('future.no_silent_downcasting', True)  
        #dfac = dfac.fillna(-1)
        dfac = dfac.round(3)
        dfac.to_csv(os.path.join(csv_dir, f'{self.VID}_cell_data.csv'), sep=';', index=False, decimal=',')

        df = pd.DataFrame(results)
        df = df.sort_values(by='name', ascending=True)
        df = df.round(3)
        df.to_csv(os.path.join(csv_dir, f'{self.VID}_image_data.csv'), sep=';', index=False, decimal=',')


        unique_vid_well_subset = df['VID_Well_Subset'].unique()
        print(unique_vid_well_subset)

        plot_dir = os.path.join(self.main_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True) 

        # Parameters for the multiplot 
        num_plots = len(unique_vid_well_subset)

        plots_per_row = 3 if num_plots < 10 else 5
        min_rows = 1
        
        # Calculate rows and columns for the multiplot
        num_rows = max(min_rows, int(np.ceil(num_plots / plots_per_row)))  
        num_cols = min(plots_per_row, num_plots)
        

        def create_single_plot(axes, subset, data_columns, title_prefix, ylabel):
            subset_data = df[df['VID_Well_Subset'] == subset]
            alive_counts = subset_data[data_columns[0]]
            dead_counts = subset_data[data_columns[1]]
            time_axis = range(len(alive_counts))
               
            axes.plot(time_axis, alive_counts, marker='o', color='blue', label='Alive')
            axes.plot(time_axis, dead_counts, marker='s', color='magenta', label='Dead')

            if len(data_columns) == 3:
                total_counts = subset_data[data_columns[2]]
                axes.plot(time_axis, total_counts, marker='*', color='orange', label='Total')

            axes.set_title(f'VID{subset}')
            axes.set_xlabel('Time (hours)')
            axes.set_ylabel(ylabel) 
            axes.legend()


        def create_multi_plot(axes, subset, data_columns, title_prefix, ylabel):
            subset_data = df[df['VID_Well_Subset'] == subset]
            alive_counts = subset_data[data_columns[0]]
            dead_counts = subset_data[data_columns[1]]
            time_axis = range(len(alive_counts))
        
            row = i // plots_per_row
            col = i % plots_per_row
        
            axes[row, col].plot(time_axis, alive_counts, marker='o', color='blue', label='Alive')
            axes[row, col].plot(time_axis, dead_counts, marker='s', color='magenta', label='Dead')

            if len(data_columns) == 3:
                total_counts = subset_data[data_columns[2]]
                axes[row, col].plot(time_axis, total_counts, marker='*', color='orange', label='Total')

            axes[row, col].set_title(f'VID{subset}')
            axes[row, col].set_xlabel('Time (hours)')
            axes[row, col].set_ylabel(ylabel) 
            axes[row, col].legend()
        
        # Create the count plots
        fig_size_x = 5 if num_plots == 1 else int(num_cols*5)
        fig_size_y = 5 if num_plots == 1 else int(num_rows*5)

        fig_conf, axes_conf = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_conf.canvas.manager.window.title('Confluence')

        fig_count, axes_count = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_count.canvas.manager.window.title('Cell Count')

        fig_avg_area, axes_avg_area = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_avg_area.canvas.manager.window.title('Average Cell Area')

        for fcn in self.flourescence_channel_names_found:

            fig_fsc, axes_fsc = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
            fig_fsc.canvas.manager.window.title(f'Normalized Total {fcn} Flourescence Intensity')

            if num_plots == 1: 
                create_single_plot(axes_fsc, unique_vid_well_subset[0], [f'alive_{fcn}_intensity', f'dead_{fcn}_intensity'], '___', f'Normalized Total {fcn} Flourescence Intensity')

            else:
                for i, subset in enumerate(unique_vid_well_subset):
                    create_multi_plot(axes_fsc, subset, [f'alive_{fcn}_intensity', f'dead_{fcn}_intensity'], '___', f'Normalized Total {fcn} Flourescence Intensity')


            try:
                fig_fsc.savefig(os.path.join(plot_dir, f'Normalized Total {fcn} Flourescence Intensity.pdf'))
            except Exception as e:
                print(f"Error saving plot: {e}")

            fig_fsc.tight_layout()


        #if axes.ndim == 1:
        if num_plots == 1:
            create_single_plot(axes_conf, unique_vid_well_subset[0], ['confluence alive', 'confluence dead', 'confluence'], '___', '%')
            create_single_plot(axes_count, unique_vid_well_subset[0], ['alive cell count', 'dead cell count', 'total cell count'], '___', 'Count')
            create_single_plot(axes_avg_area, unique_vid_well_subset[0], ['avg_alive_area', 'avg_dead_area'], '___', 'Average Cell Area in Pixel')
        
        else:
            for i, subset in enumerate(unique_vid_well_subset):
                create_multi_plot(axes_conf, subset, ['confluence alive', 'confluence dead', 'confluence'], '___', '%')
                create_multi_plot(axes_count, subset, ['alive cell count', 'dead cell count', 'total cell count'], '___', 'Count')
                create_multi_plot(axes_avg_area, subset, ['avg_alive_area', 'avg_dead_area'], '___', 'Average Cell Area in Pixel')
        

        try:
            fig_conf.savefig(os.path.join(plot_dir, 'confluence.pdf'))
            fig_count.savefig(os.path.join(plot_dir, 'Cell Count.pdf'))
            fig_avg_area.savefig(os.path.join(plot_dir, 'Average Cell Area in Pixel.pdf'))
        except Exception as e:
            print(f"Error saving plot: {e}")


        fig_conf.tight_layout()
        fig_count.tight_layout()
        fig_avg_area.tight_layout()
        
        plt.show()



    def analyze_cell_flourescence_channels(self, cells, flourescence_channels, avg_size_alive, avg_size_dead):

        if flourescence_channels:
    
            img_h, img_w = flourescence_channels[0].org_color.shape[:2]
    
            for cell in cells:
                x = int(cell.x)
                y = int(cell.y)

                if cell.state == 'alive':
                    size = avg_size_alive
                else:
                    size = avg_size_dead

                avg_flourrescence_intesities = {}

                for fscc in flourescence_channels:

                    if (x >= size) and (x < (img_w - size - 1)) and (y >= size) and (y < (img_h - size - 1)):

                        fav = flourescence_average_values(-1,-1,-1)

                        if fscc.org_gray is not None:
                            org_gray_crop = fscc.org_gray[int(y-size):int(y+size), int(x-size):int(x+size)]
                            average_org_gray = np.mean(org_gray_crop)
                            fav.org = average_org_gray

                        if fscc.normalized_gray is not None:
                            normalized_gray_crop = fscc.normalized_gray[int(y-size):int(y+size), int(x-size):int(x+size)]
                            average_normalized_gray = np.mean(normalized_gray_crop)
                            fav.normalized = average_normalized_gray

                        if fscc.denoised_gray is not None:
                            denoised_gray_crop = fscc.denoised_gray[int(y-size):int(y+size), int(x-size):int(x+size)]
                            average_denoised_gray = np.mean(denoised_gray_crop)
                            fav.denoised = average_denoised_gray

                        avg_flourrescence_intesities[fscc.name] = fav

                    else:
                        fav = flourescence_average_values(-2,-2,-2)
                        avg_flourrescence_intesities[fscc.name] = fav

                cell.avg_flourrescence_intesities = avg_flourrescence_intesities
                               
            return cells



    def process_single_image(self, image_path):
        global global_model
        input_image = cv2.imread(image_path)
        input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        height, width = input_image_gray.shape[:2]

        crops, num_cutouts_w, num_cutouts_h = create_crops(input_image_gray, crop_size, crop_border, height, width)

        preds = []

        for crop in crops:

            prepared_crop = np.array(crop.reshape(-1, crop_size, crop_size, 1), dtype=np.float32)
                            
            global_model.set_tensor(global_model.get_input_details()[0]['index'], prepared_crop)
            global_model.invoke()
                            
            output_data_tflite = global_model.get_tensor(global_model.get_output_details()[0]['index'])

            preds.append(output_data_tflite[0])


        full_img_preds = restore_image_from_crops(preds, crop_size, crop_border, num_cutouts_w, num_cutouts_h, height, width)
                                
        alive, dead, pos = cv2.split(full_img_preds)

        alive_uint = ((alive > 0.5)*255).astype(np.uint8)
        dead_uint = ((dead > 0.5)*255).astype(np.uint8)
        pos_uint = ((pos > 0.5)*255).astype(np.uint8)      
        conf_uint = np.logical_or(alive_uint, dead_uint).astype(np.uint8) * 255 

        alive_conf = (np.count_nonzero(alive_uint) / alive_uint.size) * 100
        dead_conf = (np.count_nonzero(dead_uint) / dead_uint.size) * 100
        total_conf = alive_conf + dead_conf

        modelname = str(config['DEFAULT']['modelname'])
        bf_dir = os.path.dirname(image_path)
        main_dir = os.path.dirname(bf_dir)
        alive_dir = os.path.join(main_dir, modelname, 'alive')
        dead_dir = os.path.join(main_dir, modelname, 'dead')
        pos_org_dir = os.path.join(main_dir, modelname, 'pos_org')
        pos_dir = os.path.join(main_dir, modelname, 'pos')
        combi_dir = os.path.join(main_dir, modelname, 'combi')

        os.makedirs(alive_dir, exist_ok=True)
        os.makedirs(dead_dir, exist_ok=True)
        os.makedirs(pos_org_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(combi_dir, exist_ok=True)
                        
        base_image_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(alive_dir, f'{base_image_name[:-4]}.png'), alive_uint)
        cv2.imwrite(os.path.join(dead_dir, f'{base_image_name[:-4]}.png'), dead_uint)
        cv2.imwrite(os.path.join(pos_org_dir, f'{base_image_name[:-4]}.png'), pos_uint)


        positions = get_pos_contours(pos_uint)
        
        pos_color = np.zeros((height,width,3), np.uint8) 
        
        for p in positions:
            cv2.circle(pos_color, p, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(pos_dir, f'{base_image_name[:-4]}_.png'), pos_color)
        
        alive_rgba = gray2rgba_mask(alive_uint, [255, 0, 0])
        dead_rgba = gray2rgba_mask(dead_uint, [136, 0, 136])
        #pos_rgba = color2rgba_mask(pos_color, [255, 255, 255])
        
        black_background = np.zeros((height,width,3), np.uint8)
        comb_phaseimg_black = cv2.addWeighted(input_image,0.7,black_background,0.3,0)
        
        comb_img1 = cv2.addWeighted(alive_rgba,1,dead_rgba,1,0)
        b, g, r, alpha = cv2.split(comb_img1)
        rgb = [b,g,r]
        comb_img1_bgr = cv2.merge(rgb,3)
        
        comb_img = cv2.addWeighted(comb_phaseimg_black,0.6,comb_img1_bgr,0.4,0)
        comb_img_pos = cv2.add(comb_img,pos_color)
        cv2.imwrite(os.path.join(combi_dir, f'{base_image_name[:-4]}.png'), comb_img_pos)
        
        
        alive_count, dead_count, unclear_count, analysed_cells = get_cell_state(positions, alive_uint, dead_uint)
        
        
        avg_alive_area = 0
        if alive_count > 0:
            avg_alive_area = round((np.sum(alive_uint)/255)/alive_count,0)
        
        avg_dead_area = 0
        if dead_count > 0:
            avg_dead_area = round((np.sum(dead_uint)/255)/dead_count,0)


        flourescence_channels = []
        alive_norm_filt_tot_ints = {}
        dead_norm_filt_tot_ints = {}
        alive_area = np.count_nonzero(alive_uint) / alive_uint.size
        dead_area = np.count_nonzero(dead_uint) / dead_uint.size



        for fcn in self.flourescence_channel_names_found:
            full_path = os.path.join(self.main_dir, fcn)
            if os.path.isdir(full_path):
                org_img = cv2.imread(os.path.join(full_path, base_image_name), cv2.IMREAD_UNCHANGED)

                dtype = org_img.dtype

                if dtype == 'uint8':
                    if len(org_img.shape) == 2:
                        img_gray = org_img
                    else:
                        img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

                elif dtype == 'uint16':
                    print("Image is 16-bit")
                    img_gray = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(os.path.join(full_path, base_image_name), img_gray) 


                elif dtype == 'float32':
                    print("Image is 32-bit")
                    img_gray = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imwrite(os.path.join(full_path, base_image_name), img_gray) 

                else:
                    print("Unrecognized bit depth")


                average_brightness = np.mean(img_gray)
                normalized_img_gray = img_gray - average_brightness
                normalized_img_uint8 = np.clip(normalized_img_gray, 0, 255).astype(np.uint8)


                if alive_area > 0:
                    norm_filt_alive = normalized_img_uint8.copy()
                    norm_filt_alive[alive_uint <= 0] = 0
                    alive_norm_filt_tot_int = np.sum(norm_filt_alive)
                    alive_norm_filt_avg_int = alive_norm_filt_tot_int / alive_area
                    alive_norm_filt_tot_ints[fcn] = alive_norm_filt_avg_int
                else:
                    norm_filt_alive = None
                    alive_norm_filt_tot_ints[fcn] = 0

                if dead_area > 0:
                    norm_filt_dead = normalized_img_uint8.copy()
                    norm_filt_dead[dead_uint == 0] = 0
                    dead_norm_filt_tot_int = np.sum(norm_filt_dead)
                    dead_norm_filt_avg_int = dead_norm_filt_tot_int / dead_area
                    dead_norm_filt_tot_ints[fcn] = dead_norm_filt_avg_int
                else:
                    norm_filt_dead = None
                    dead_norm_filt_tot_ints[fcn] = 0

                normalized_filtered_conf = normalized_img_uint8.copy()
                normalized_filtered_conf[conf_uint] = 0

                fcci = flourescence_channel_image(fcn, org_img, img_gray, 
                                                  normalized_img_uint8, norm_filt_alive, norm_filt_dead, normalized_filtered_conf,
                                                  None, None, None, None)

                flourescence_channels.append(fcci)

                #filtered_alive = normalized_img_gray.copy()
                #filtered_alive[alive_uint == 0] = 0
                #cv2.imwrite(f'E://PyInstallerTests/flourescence_state_filtered/alive_{base_image_name[:-4]}.png', filtered_alive)
                #
                #filtered_dead = normalized_img_gray.copy()
                #filtered_dead[dead_uint == 0] = 0
                #cv2.imwrite(f'E://PyInstallerTests/flourescence_state_filtered/dead_{base_image_name[:-4]}.png', filtered_dead)

                #filtered_conf = normalized_img_gray.copy()
                #filtered_conf[conf_uint] = 0
                #cv2.imwrite(f'E://PyInstallerTests/flourescence_state_filtered/conf_{base_image_name[:-4]}.png', filtered_conf)


        avg_alive_size = np.floor(np.sqrt(avg_alive_area))
        avg_dead_size = np.floor(np.sqrt(avg_dead_area))

        analysed_cells_final = self.analyze_cell_flourescence_channels(analysed_cells, flourescence_channels, avg_alive_size, avg_dead_size)

        image_name_parts = base_image_name[:-4].split("_")

        result_dict = {
            'name' : base_image_name[:-4],
            'VID_Well_Subset' : f'{image_name_parts[0][3:]}_{image_name_parts[1]}_{image_name_parts[2]}',
            'VID' : image_name_parts[0][3:],
            'Well' : image_name_parts[1],
            'Subset' : image_name_parts[2],
            'total cell count' : alive_count+dead_count,
            'alive cell count' : alive_count,
            'dead cell count' : dead_count,
            'confluence' : total_conf,
            'confluence alive' : alive_conf,
            'confluence dead' : dead_conf,
            'avg_alive_area' : avg_alive_area,
            'avg_dead_area' : avg_dead_area,
            }


        for color, intensity in alive_norm_filt_tot_ints.items():
            result_dict[f'alive_{color}_intensity'] = intensity

        for color, intensity in dead_norm_filt_tot_ints.items():
            result_dict[f'dead_{color}_intensity'] = intensity


        cells_list = []  # Create an empty list to store cell dictionaries

        for ac in analysed_cells_final:
            # Create a temporary dictionary for each cell
            cell_dict = {
                'name': base_image_name[:-4],  # Assuming base_image_name is defined elsewhere
                'x': ac.x,
                'y': ac.y,
                'state': ac.state
            }
            
            # Loop through avg_flourrescence_intesities and add them to the cell_dict
            for color, int_values in ac.avg_flourrescence_intesities.items():

                cell_dict[f'{color}_org'] = int_values.org
                cell_dict[f'{color}_norm'] = int_values.normalized
                cell_dict[f'{color}_denoised'] = int_values.denoised
            
            # Append the cell_dict to the cells_list
            cells_list.append(cell_dict)
        
        # Convert the list of dictionaries to a pandas DataFrame
        cells_df = pd.DataFrame(cells_list)

        


        return (result_dict, cells_df)



   











def on_resource_dropdown_change(event_value):

    config['DEFAULT']['resource_allocation'] = event_value
    
    # Write the updated configuration back to the file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)                     # Save to config.ini 



def open_settings():
    print("Settings")

    new_window = tk.Toplevel()
    new_window.title("Settings")
    new_window.geometry("550x270")
    new_window.lift()
    new_window.attributes('-topmost', True)

    

    # Frame for model selection
    denoiser_model_frame = tk.Frame(new_window)
    denoiser_model_frame.grid(row=0, column=0, sticky="nsew")

    modelname = str(config['DEFAULT']['modelname'])
    
    analyser_model_path_var = tk.StringVar(new_window, value=modelname)

    def open_analyser_model_dialog():
        new_model_path = filedialog.askopenfilename(filetypes=[("TFlite Models", "*.tflite")])

        if new_model_path:
            (config['DEFAULT']['model_path']) = new_model_path
            new_modelname_ext = os.path.basename(new_model_path)
            new_modelname, _ = os.path.splitext(new_modelname_ext)
            config['DEFAULT']['modelname'] = new_modelname
            analyser_model_path_var.set(new_modelname)

    analyser_model_frame = tk.Frame(new_window)
    analyser_model_frame.grid(row=1, column=0, sticky="nsew")

    select_model_button = ctk.CTkButton(analyser_model_frame, text="Select Analyser Model...", command=open_analyser_model_dialog)
    select_model_button.pack(side="left", pady=10, padx=5)
    
    model_analyser_label = ctk.CTkLabel(analyser_model_frame, textvariable=analyser_model_path_var)
    model_analyser_label.pack(side="left", pady=10, padx=5)

    # Frame for checkboxes 
    checkbox_frame = tk.Frame(new_window)
    checkbox_frame.grid(row=2, column=0, sticky="nsew")

    checkbox2 = ctk.CTkCheckBox(checkbox_frame, text="Save image with overlay?")
    checkbox2.pack(pady=(10,50), padx=5, anchor='w')


    save_close_frame = tk.Frame(new_window)
    save_close_frame.grid(row=3, column=0, sticky="nsew")

    def save_settings():
        with open('config.ini', 'w') as configfile:
            config.write(configfile)




    save_button = ctk.CTkButton(save_close_frame, text="Save", command=save_settings, fg_color="green")
    save_button.pack(side="left", pady=10, padx=5)

    close_button = ctk.CTkButton(save_close_frame, text="Close", command=new_window.destroy, fg_color="red")
    close_button.pack(side="left", pady=10, padx=5)



def main(): 
    root = ctk.CTk()  # Create a customtkinter root window
    root.title("Cellanalyser Lite")
    root.geometry('400x200')

    CA = CellAnalyzer()

    menubar = Menu(root)

    def open_manual():
        # Get the absolute path of the PDF file (for platform compatibility)
        pdf_path = os.path.abspath('CellLocator_Manual.pdf')

        # Attempt to open the manual
        try:
            os.startfile(pdf_path)  # OS-appropriate way to open files
        except OSError:
            print("Error: Could not open PDF file. Check if it exists and you have a PDF reader.") 


    def open_github():
        github_url = "https://github.com/MichaelVorndran"  # Replace with your actual URL
        webbrowser.open(github_url)


    # Settings
    settings_menu = Menu(menubar, tearoff=0)
    settings_menu.add_command(label="Settings", command=open_settings)
    settings_menu.add_command(label="Manual", command=open_manual)
    settings_menu.add_command(label="GitHub", command=open_github)
    settings_menu.add_command(label="Citeation", command=open_settings)
    menubar.add_cascade(label="Menu", menu=settings_menu)

    root.config(menu=menubar)


    # Resource Selection Dropdown 
    selected_value = config['DEFAULT']['resource_allocation']
    resource_options = ["25", "50", "75", "100"] 
    resource_label = ctk.CTkLabel(root, text="CPU Resource Allocation (%)")  
    resource_label.pack()

    resource_dropdown = ctk.CTkComboBox(master=root, values=resource_options, command=on_resource_dropdown_change)
    resource_dropdown.set(selected_value)
    resource_dropdown.pack()

    # Buttons 
    btn_open = ctk.CTkButton(root, text="Select Phase Image Directory", command=CA.on_select_bf_dir_clicked)
    btn_open.configure(width=220)
    btn_open.pack(pady=10)

    btn_analyse = ctk.CTkButton(root, text="Analyse", command=CA.on_analyse_clicked)
    btn_analyse.configure(width=220)
    btn_analyse.pack(pady=10)


    root.mainloop()

if __name__ == '__main__':  # Main guard
    multiprocessing.freeze_support()  
    main() 
