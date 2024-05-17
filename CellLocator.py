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
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0
import matplotlib.backends.backend_pdf
import math
from tqdm import tqdm
import configparser
import webbrowser
import gc

from mpl_toolkits.mplot3d import Axes3D 
import plotly.express as px
import plotly.graph_objects as go



global_model = None


config = configparser.ConfigParser() 
config.read('config.ini')

resource_allocation = int(config['DEFAULT']['resource_allocation'])
crop_size = int(config['DEFAULT']['crop_size'])
crop_border = int(config['DEFAULT']['crop_border'])
magnification = str(config['DEFAULT']['magnification'])


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


def create_color_map(R_max, G_max, B_max, max_grayscale):
    """
    Creates a color map from RGB to grayscale mapping.
    
    Args:
    R_max (int): Maximum value of the Red component in the RGB model
    G_max (int): Maximum value of the Green component in the RGB model
    B_max (int): Maximum value of the Blue component in the RGB model
    max_grayscale (int): Maximum grayscale value that corresponds to (R_max, G_max, B_max)
    
    Returns:
    dict: A dictionary mapping each grayscale value to an RGB tuple
    """
    color_map = {}
    for grayscale_val in range(max_grayscale + 1):
        R = int((R_max / max_grayscale) * grayscale_val)
        G = int((G_max / max_grayscale) * grayscale_val)
        B = int((B_max / max_grayscale) * grayscale_val)
        color_map[grayscale_val] = (R, G, B)
    return color_map


def colorize_with_mapping(grayscale_image, color_map, max_grayscale_value):
    
    h, w = grayscale_image.shape[:2]

    grayscale_image[grayscale_image>=max_grayscale_value] = max_grayscale_value

    colorized_image = np.zeros((h,w,3), dtype=np.uint8)

    for key in color_map:
        colorized_image[grayscale_image==key] = color_map[key]

    return colorized_image












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

def restore_image_from_crops(crops, crop_size, crop_border, output_image_channels, num_cutouts_w, num_cutouts_h, org_h, org_w):

    h = int(num_cutouts_h * (crop_size-crop_border))
    w = int(num_cutouts_w * (crop_size-crop_border))
    inner_crop_size = int(crop_size - 2*crop_border)

    background = np.zeros((h,w,output_image_channels), np.float32) 


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

        self.model_path_analyser = str(config['DEFAULT']['model_path_analyser'])
        self.model_path_denoiser = str(config['DEFAULT']['model_path_denoiser'])
        self.modelname_analyser = ''
        self.modelname_denoiser = ''

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

        config = configparser.ConfigParser()
        config.read('config.ini')

        use_denoiser = config['DEFAULT'].getboolean('use_denoiser') 

        if use_denoiser:
            for fscn in self.flourescence_channel_names_found:
                fscn_dir = os.path.join(self.main_dir, fscn)

                if fscn == 'red':
                    self.max_grayscale_value = int(0.114*0 + 0.587*39 + 0.299*255)+1
                    self.color_map = create_color_map(0, 39, 255, self.max_grayscale_value)

                elif fscn == 'orange':
                    self.max_grayscale_value = int(0.114*0 + 0.587*0 + 0.299*255)+1
                    self.color_map = create_color_map(0, 0, 255, self.max_grayscale_value)
                    
                elif fscn == 'green':
                    self.max_grayscale_value = int(0.114*0 + 0.587*255 + 0.299*0)+1
                    self.color_map = create_color_map(0, 255, 0, self.max_grayscale_value)
                    
                elif fscn == 'nir':
                    self.max_grayscale_value = int(0.114*255 + 0.587*128 + 0.299*0)+1
                    self.color_map = create_color_map(255, 128, 0, self.max_grayscale_value)
                    
                else:
                    self.max_grayscale_value = int(0.114*0 + 0.587*39 + 0.299*255)+1
                    self.color_map = create_color_map(0, 39, 255, self.max_grayscale_value)
                    
                
                self.current_fscn = fscn

                images = [os.path.join(fscn_dir, img) for img in os.listdir(fscn_dir)]
                total_images = len(images)
                
                pbar = tqdm(total=total_images)
                
                
                
                resource_allocation = int(config['DEFAULT']['resource_allocation'])
                model_path_denoiser = str(config['DEFAULT']['model_path_denoiser'])
                
                num_processes = int(max((mp.cpu_count() * (resource_allocation / 100)),1))
                if num_processes > 1:
                    print(f"\nStarting {num_processes} processes to denoise {fscn} flourescence channel ")
                else:
                    print("\nStarting 1 process")

                with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path_denoiser,)) as pool:
                    for _ in pool.imap_unordered(self.process_single_image_denoiser, images):

                        pbar.update(1)
                
                pbar.close()
                print(f"Denoising {fscn} flourescence channel complete.")



        images = [os.path.join(self.phase_img_dir, img) for img in os.listdir(self.phase_img_dir)]
        total_images = len(images)
        
        pbar = tqdm(total=total_images)
        
        #config = configparser.ConfigParser()
        #config.read('config.ini')
        resource_allocation = int(config['DEFAULT']['resource_allocation'])
        model_path = str(config['DEFAULT']['model_path_analyser'])
        csv_decimal = str(config['DEFAULT']['csv_decimal'])
        #print(f'csv_decimal: {csv_decimal}')
        
        
        num_processes = int(max((mp.cpu_count() * (resource_allocation / 100)),1))
        if num_processes > 1:
            print("\nStarting", num_processes, " processes to analyse phase images.")
        else:
            print("\nStarting 1 process")


        self.modelname_analyser = config['DEFAULT']['modelname_analyser']
        csv_dir = os.path.join(self.main_dir, self.modelname_analyser, 'cell_data')
        os.makedirs(csv_dir, exist_ok=True) 
        print(csv_dir)
        
        results = []
        cell_results_df = []
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
            for result_tuple  in pool.imap_unordered(self.process_single_image_analysis, images):
                results_dict, analysed_cells_df = result_tuple
                results.append(results_dict)
                cell_results_df.append(analysed_cells_df)

                pbar.update(1)
        
        pbar.close()
        print("Analysis complete.")


        print('Saving data.')

        dfac = pd.concat(cell_results_df, ignore_index=True)
        dfac = dfac.round(3)
        dfac.to_csv(os.path.join(csv_dir, f'{self.VID}_cell_data.csv'), sep=';', index=False, decimal=csv_decimal)

        df = pd.DataFrame(results)
        df = df.sort_values(by='name', ascending=True)
        df = df.round(3)
        df.to_csv(os.path.join(csv_dir, f'{self.VID}_image_data.csv'), sep=';', index=False, decimal=csv_decimal)



        print('Creating Plots.')

        unique_vid_well_subsets = df['VID_Well_Subset'].unique()
        print(unique_vid_well_subsets)

        plot_dir = os.path.join(self.main_dir, self.modelname_analyser, 'plots')
        os.makedirs(plot_dir, exist_ok=True) 

        # Parameters for the multiplot 
        num_plots = len(unique_vid_well_subsets)

        # get num of subsets

        def extract_wellname(text):
            start_index = text.find("_") + 1
            end_index = text.rfind("_")
            return text[start_index:end_index]

        #unique_wells = set(extract_wellname(item) for item in unique_vid_well_subsets)

        well_counts = {}
        for item in unique_vid_well_subsets:

          wellname = extract_wellname(item)

          if wellname in well_counts:
            well_counts[wellname] += 1
          else:
            well_counts[wellname] = 1


        max_num_subsets_per_well = max(well_counts.values())
        #print(f'max_num_subsets_per_well: {max_num_subsets_per_well}')

        if num_plots <= 2:
            plots_per_row = 1
        elif num_plots <= 4:
            plots_per_row = 2
        else:
            plots_per_row = max(2, max_num_subsets_per_well)


        min_rows = 1
        
        # Calculate rows and columns for the multiplot
        num_rows = max(min_rows, int(np.ceil(num_plots / plots_per_row)))  
        num_cols = min(plots_per_row, num_plots)

        print(f'num_rows: {num_rows}')
        print(f'num_cols: {num_cols}')
       

        def create_single_plot(axes, vid_well_subset, data_columns, ylabel):
            subset_data = df[df['VID_Well_Subset'] == vid_well_subset]

            time_axis = range(len(subset_data)) 

            colors = ['blue', 'magenta', 'orange']
            labels = ['Alive', 'Dead', 'Total']
            markers = ['o', 's', '*']
            
            for i, col_name in enumerate(data_columns):
                if len(data_columns)==1:
                    label = 'Confluence'
                    i=2
                else:
                    label = labels[i]

                data_column = subset_data[col_name]
                axes.plot(time_axis, data_column, marker=markers[i], color=colors[i], label=label)

            axes.set_title(f'VID{vid_well_subset}')
            axes.set_xlabel('Time')
            axes.set_ylabel(ylabel)
            axes.set_ylim(bottom=0) 
            axes.legend()


        def create_multi_plot(axes, vid_well_subset, data_columns, ylabel, i):

            row = i // num_cols
            col = i % num_cols

            if num_cols == 1:
                ax = axes[row]
            elif num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            subset_data = df[df['VID_Well_Subset'] == vid_well_subset]

            time_axis = range(len(subset_data)) 

            colors = ['blue', 'magenta', 'orange']
            labels = ['Alive', 'Dead', 'Total']
            markers = ['o', 's', '*']
            
            for i, col_name in enumerate(data_columns):
                if len(data_columns)==1:
                    label = 'Confluence'
                    i=2
                else:
                    label = labels[i]

                data_column = subset_data[col_name]
                ax.plot(time_axis, data_column, marker=markers[i], color=colors[i], label=label)

            ax.set_title(f'VID{vid_well_subset}')
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0) 
            ax.legend()


        
        # Create the count plots
        fig_size_x = 5 if num_plots == 1 else int(num_cols*5)
        fig_size_y = 5 if num_plots == 1 else int(num_rows*5)

        fig_conf, axes_conf = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_conf.canvas.manager.window.title('Confluence')

        fig_conf_cellocate, axes_conf_cellocate = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_conf_cellocate.canvas.manager.window.title('Confluence_Cellocate')

        fig_count, axes_count = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_count.canvas.manager.window.title('Cell Count')

        fig_avg_area, axes_avg_area = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
        fig_avg_area.canvas.manager.window.title('Average Cell Area')


        use_denoiser = config['DEFAULT'].getboolean('use_denoiser')  

        flourescence_channel_typs = []
        flourescence_channel_typs.append('norm')

        if use_denoiser:
            flourescence_channel_typs.append('denoised')

        for fct in flourescence_channel_typs:

            plot_data = fct

            if fct == 'norm':
                plot_label = 'Normalized'
            elif fct == 'denoised':
                plot_label = 'Denoised'


            if len(self.flourescence_channel_names_found) == 2:

                fcns = []
                for fcn in self.flourescence_channel_names_found:
                    fcns.append(fcn)
        
                color_map = {'alive': 'blue', 'dead': 'magenta'}
                #colors = dfac['state'].map(color_map)
        
                for i, subset in enumerate(unique_vid_well_subsets):

                    subset_df = dfac[dfac['name'].str.contains(subset)]
                    subset_df = subset_df.sort_values(by='name', ascending=True)  

                    # Create 2D scatter plot
                    plotly_fig = px.scatter(subset_df, x=f'{fcns[0]}_{plot_data}', y=f'{fcns[1]}_{plot_data}', 
                                             animation_frame='name', color='state',
                                             color_discrete_map=color_map,
                                             labels={'fcns[0]_{plot_data}': f"{fcns[0]} {plot_label}", 
                                                     'fcns[1]_{plot_data}': f"{fcns[1]} {plot_label}"},
                                             title="Animated 2D Scatter Plot")

                    plotly_fig.update_traces(marker=dict(size=3))

                    # Update layout for 2D
                    plotly_fig.update_layout(xaxis=dict(range=[0, subset_df[f'{fcns[0]}_{plot_data}'].max()], 
                                                         title=f'{fcns[0]} {plot_label}'),
                                             yaxis=dict(range=[0, subset_df[f'{fcns[1]}_{plot_data}'].max()], 
                                                        title=f'{fcns[1]} {plot_label}'))

                    plotly_fig.write_html(os.path.join(plot_dir, f'{plot_data}_{subset}.html'))
                    #plotly_fig.show()


            if len(self.flourescence_channel_names_found) == 3:
        
                fcns = []
                for fcn in self.flourescence_channel_names_found:
                    fcns.append(fcn)
        
                color_map = {'alive': 'blue', 'dead': 'magenta'}
                #colors = dfac['state'].map(color_map)
        
                for i, subset in enumerate(unique_vid_well_subsets):

                    subset_df = dfac[dfac['name'].str.contains(subset)]
                    subset_df = subset_df.sort_values(by='name', ascending=True)  
        
                    plotly_fig = px.scatter_3d(subset_df, x=f'{fcns[0]}_{plot_data}', y=f'{fcns[1]}_{plot_data}', z=f'{fcns[2]}_{plot_data}',
                            animation_frame='name', color='state',
                            color_discrete_map=color_map,
                            labels={'green_{plot_data}': f"{fcns[0]} {plot_label}", 'orange_{plot_data}': f"{fcns[1]} {plot_label}", 'nir_{plot_data}': f"{fcns[2]} {plot_label}"},
                            title="Animated 3D Scatter Plot")
        
                    plotly_fig.update_traces(marker=dict(size=3))
        
                    plotly_fig.update_layout(scene=dict(
                                    xaxis=dict(range=[0, subset_df[f'{fcns[0]}_{plot_data}'].max()], title=f'{fcns[0]} {plot_label}'),
                                    yaxis=dict(range=[0, subset_df[f'{fcns[1]}_{plot_data}'].max()], title=f'{fcns[1]} {plot_label}'),
                                    zaxis=dict(range=[0, subset_df[f'{fcns[2]}_{plot_data}'].max()], title=f'{fcns[2]} {plot_label}'),
                                    aspectratio=dict(x=1, y=1, z=1),
                                    aspectmode='manual'
                                ),
                                    margin=dict(l=0, r=0, b=0, t=0)
                                )



                    plotly_fig.write_html(os.path.join(plot_dir, f'{plot_data}_{subset}.html'))
                    #plotly_fig.show()



            for fcn in self.flourescence_channel_names_found:

                fig_fsc, axes_fsc = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
                fig_fsc.canvas.manager.window.title(f'{plot_label} Total {fcn} Flourescence Intensity')

                if num_plots == 1: 
                    create_single_plot(axes_fsc, unique_vid_well_subsets[0], [f'alive_{fcn}_{fct}_intensity', f'dead_{fcn}_{fct}_intensity'], f'{plot_label} Total {fcn} Flourescence Intensity')

                else:
                    for i, vid_well_subset in enumerate(unique_vid_well_subsets):
                        #print(fcn)
                        create_multi_plot(axes_fsc, vid_well_subset, [f'alive_{fcn}_{fct}_intensity', f'dead_{fcn}_{fct}_intensity'], f'{plot_label} Total {fcn} Flourescence Intensity', i)


                try:
                    fig_fsc.savefig(os.path.join(plot_dir, f'{plot_label} Total {fcn} Flourescence Intensity.pdf'))
                except Exception as e:
                    print(f"Error saving plot: {e}")

                fig_fsc.tight_layout()


            for fcn in self.flourescence_channel_names_found:

                fig_fsc, axes_fsc = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_size_x, fig_size_y))
                fig_fsc.canvas.manager.window.title(f'{plot_label} Mean {fcn} Flourescence Intensity')

                if num_plots == 1: 
                    create_single_plot(axes_fsc, unique_vid_well_subsets[0], [f'mean_alive_{fcn}_{fct}_intensity', f'mean_dead_{fcn}_{fct}_intensity'], f'{plot_label} Mean {fcn} Flourescence Intensity')

                else:
                    for i, vid_well_subset in enumerate(unique_vid_well_subsets):
                        create_multi_plot(axes_fsc, vid_well_subset, [f'mean_alive_{fcn}_{fct}_intensity', f'mean_dead_{fcn}_{fct}_intensity'], f'{plot_label} Mean {fcn} Flourescence Intensity', i)


                try:
                    fig_fsc.savefig(os.path.join(plot_dir, f'{plot_label} Mean {fcn} Flourescence Intensity.pdf'))
                except Exception as e:
                    print(f"Error saving plot: {e}")

                fig_fsc.tight_layout()


        #if axes.ndim == 1:
        if num_plots == 1:
            create_single_plot(axes_conf, unique_vid_well_subsets[0], ['confluence'], '%')
            create_single_plot(axes_conf_cellocate, unique_vid_well_subsets[0], ['confluence alive', 'confluence dead', 'confluence'], '%')
            create_single_plot(axes_count, unique_vid_well_subsets[0], ['alive cell count', 'dead cell count', 'total cell count'], 'Count')
            create_single_plot(axes_avg_area, unique_vid_well_subsets[0], ['avg_alive_area', 'avg_dead_area'], 'Average Cell Area in $\mu m$')
        
        else:
            for i, vid_well_subset in enumerate(unique_vid_well_subsets):
                create_multi_plot(axes_conf, vid_well_subset, ['confluence'], '%', i)
                create_multi_plot(axes_conf_cellocate, vid_well_subset, ['confluence alive', 'confluence dead', 'confluence'], '%', i)
                create_multi_plot(axes_count, vid_well_subset, ['alive cell count', 'dead cell count', 'total cell count'], 'Count', i)
                create_multi_plot(axes_avg_area, vid_well_subset, ['avg_alive_area', 'avg_dead_area'], 'Average Cell Area in $\mu m$', i)
        

        try:
            fig_conf.savefig(os.path.join(plot_dir, 'confluence.pdf'))
            fig_conf_cellocate.savefig(os.path.join(plot_dir, 'confluence_cellocate.pdf'))
            fig_count.savefig(os.path.join(plot_dir, 'Cell Count.pdf'))
            fig_avg_area.savefig(os.path.join(plot_dir, 'Average Cell Area.pdf'))
        except Exception as e:
            print(f"Error saving plot: {e}")


        fig_conf.tight_layout()
        fig_count.tight_layout()
        fig_avg_area.tight_layout()
        
        #plt.show()

        print('all done')



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




    def process_single_image_denoiser(self, image_path):
        global global_model

        org_img = cv2.imread(os.path.join(image_path), cv2.IMREAD_UNCHANGED)
        
        dtype = org_img.dtype
        
        if dtype == 'uint8':
            if len(org_img.shape) == 2:
                input_image_gray = org_img
            else:
                input_image_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        
        elif dtype == 'uint16':
            input_image_gray = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        elif dtype == 'float32':
            input_image_gray = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        else:
            print("Unrecognized bit depth")


        height, width = input_image_gray.shape[:2]

        crops, num_cutouts_w, num_cutouts_h = create_crops(input_image_gray, crop_size, crop_border, height, width)

        preds = []

        for crop in crops:

            prepared_crop = np.array(crop.reshape(-1, crop_size, crop_size, 1), dtype=np.float32)
                            
            global_model.set_tensor(global_model.get_input_details()[0]['index'], prepared_crop)
            global_model.invoke()
                            
            output_data_tflite = global_model.get_tensor(global_model.get_output_details()[0]['index'])

            preds.append(output_data_tflite[0])


        full_img_pred = restore_image_from_crops(preds, crop_size, crop_border, 1, num_cutouts_w, num_cutouts_h, height, width)
        full_img_pred_uint = np.squeeze(full_img_pred.astype(np.uint8))

        #print(full_img_pred_uint.shape)
        # restore color
        if len(org_img.shape) == 3:
            full_img_pred_uint = colorize_with_mapping(full_img_pred_uint, self.color_map, self.max_grayscale_value)


        out_dir = os.path.join(self.main_dir, f'{self.current_fscn}_denoised')
        os.makedirs(out_dir, exist_ok=True)

        base_image_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(out_dir, base_image_name), full_img_pred_uint)



    def process_single_image_analysis(self, image_path):
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


        full_img_preds = restore_image_from_crops(preds, crop_size, crop_border, 4, num_cutouts_w, num_cutouts_h, height, width)
                                
        conf, alive, dead, pos = cv2.split(full_img_preds)

        conf_uint = ((conf > 0.5)*255).astype(np.uint8)
        alive_uint = ((alive > 0.5)*255).astype(np.uint8)
        dead_uint = ((dead > 0.5)*255).astype(np.uint8)
        pos_uint = ((pos > 0.5)*255).astype(np.uint8)      
        #conf_uint = np.logical_or(alive_uint, dead_uint).astype(np.uint8) * 255 

        alive_conf = (np.count_nonzero(alive_uint) / alive_uint.size) * 100
        dead_conf = (np.count_nonzero(dead_uint) / dead_uint.size) * 100
        #total_conf = alive_conf + dead_conf
        total_conf = (np.count_nonzero(conf_uint) / conf_uint.size) * 100

        modelname = str(config['DEFAULT']['modelname_analyser'])
        bf_dir = os.path.dirname(image_path)
        main_dir = os.path.dirname(bf_dir)
        conf_dir = os.path.join(main_dir, modelname, 'conf')
        alive_dir = os.path.join(main_dir, modelname, 'alive')
        dead_dir = os.path.join(main_dir, modelname, 'dead')
        pos_org_dir = os.path.join(main_dir, modelname, 'pos_org')
        pos_dir = os.path.join(main_dir, modelname, 'pos')
        cellocate_overlay_dir = os.path.join(main_dir, modelname, 'cellocate_overlay')
        conf_overlay_dir = os.path.join(main_dir, modelname, 'conf_overlay')

        os.makedirs(conf_dir, exist_ok=True)
        os.makedirs(alive_dir, exist_ok=True)
        os.makedirs(dead_dir, exist_ok=True)
        os.makedirs(pos_org_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(cellocate_overlay_dir, exist_ok=True)
        os.makedirs(conf_overlay_dir, exist_ok=True)
                        
        base_image_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(conf_dir, f'{base_image_name[:-4]}.png'), conf_uint)
        cv2.imwrite(os.path.join(alive_dir, f'{base_image_name[:-4]}.png'), alive_uint)
        cv2.imwrite(os.path.join(dead_dir, f'{base_image_name[:-4]}.png'), dead_uint)
        cv2.imwrite(os.path.join(pos_org_dir, f'{base_image_name[:-4]}.png'), pos_uint)


        positions = get_pos_contours(pos_uint)
        
        pos_color = np.zeros((height,width,3), np.uint8) 
        
        for p in positions:
            cv2.circle(pos_color, p, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(pos_dir, f'{base_image_name[:-4]}_.png'), pos_color)
        
        conf_rgb = gray2rgba_mask(conf_uint, [0, 255, 255])
        alive_rgba = gray2rgba_mask(alive_uint, [255, 0, 0])
        dead_rgba = gray2rgba_mask(dead_uint, [255, 0, 255])
        
        comb_img1 = cv2.addWeighted(alive_rgba,1,dead_rgba,1,0)
        b, g, r, alpha = cv2.split(comb_img1)
        rgb = [b,g,r]
        comb_img_rgb = cv2.merge(rgb,3)

        comb_img1 = cv2.addWeighted(alive_rgba,1,dead_rgba,1,0)
        b, g, r, alpha = cv2.split(conf_rgb)
        rgb = [b,g,r]
        conf_img_rgb = cv2.merge(rgb,3)

        new_image = np.zeros(input_image.shape, input_image.dtype)
        new_image = cv2.convertScaleAbs(input_image, alpha=1.5, beta=0)
        
        cellocate_overlay_img = cv2.addWeighted(new_image,0.8,comb_img_rgb,0.8,0)
        cellocate_overlay_img_pos = cv2.add(cellocate_overlay_img,pos_color)
        cv2.imwrite(os.path.join(cellocate_overlay_dir, f'{base_image_name[:-4]}.png'), cellocate_overlay_img_pos)


        #print(f"new_image.shape {new_image.shape}")
        #print(f"conf_img_rgb.shape {conf_img_rgb.shape}")

        conf_overlay_img = cv2.addWeighted(new_image,0.8,conf_img_rgb,0.2,0)
        #conf_img_pos = cv2.add(comb_img,pos_color)
        cv2.imwrite(os.path.join(conf_overlay_dir, f'{base_image_name[:-4]}.png'), conf_overlay_img)
        
        
        alive_count, dead_count, unclear_count, analysed_cells = get_cell_state(positions, alive_uint, dead_uint)
        
        magnification = str(config['DEFAULT']['magnification'])

        if magnification == '4x':
            mag_factor = 2.82 
        elif  magnification == '10x':
            mag_factor = 1.24  
        elif  magnification == '20x':
            mag_factor = 0.62 

        avg_alive_area = 0
        if alive_count > 0:
            avg_alive_area = round(((np.sum(alive_uint)/255)/alive_count) * mag_factor,0)
        
        avg_dead_area = 0
        if dead_count > 0:
            avg_dead_area = round(((np.sum(dead_uint)/255)/dead_count) * mag_factor,0)


        flourescence_channels = []
        alive_norm_filt_tot_ints = {}
        dead_denoised_filt_tot_ints = {}
        alive_denoised_filt_tot_ints = {}
        dead_norm_filt_tot_ints = {}
        alive_area = (np.count_nonzero(alive_uint) / alive_uint.size) * mag_factor
        dead_area = (np.count_nonzero(dead_uint) / dead_uint.size) * mag_factor



        for fcn in self.flourescence_channel_names_found:
            full_path = os.path.join(self.main_dir, fcn)
            full_path_denoised = os.path.join(self.main_dir, f'{fcn}_denoised')

            if os.path.isdir(full_path):

                try:
                    if os.path.isfile(os.path.join(full_path, base_image_name)):
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

                    else:
                        org_img = np.zeros((height,width), np.uint8)
                        img_gray = org_img


                    average_brightness = np.mean(img_gray)
                    normalized_img_gray = img_gray - average_brightness
                    normalized_img_uint8 = np.clip(normalized_img_gray, 0, 255).astype(np.uint8)


                    if alive_area > 0:
                        norm_filt_alive = normalized_img_uint8.copy()
                        norm_filt_alive[alive_uint <= 0] = 0
                        alive_norm_filt_tot_int = np.sum(norm_filt_alive)
                        alive_norm_filt_tot_ints[fcn] = alive_norm_filt_tot_int
                        #alive_norm_filt_avg_int = alive_norm_filt_tot_int / alive_area
                        #alive_norm_filt_tot_ints[fcn] = alive_norm_filt_avg_int
                    else:
                        norm_filt_alive = None
                        alive_norm_filt_tot_ints[fcn] = 0

                    if dead_area > 0:
                        norm_filt_dead = normalized_img_uint8.copy()
                        norm_filt_dead[dead_uint == 0] = 0
                        dead_norm_filt_tot_int = np.sum(norm_filt_dead)
                        dead_norm_filt_tot_ints[fcn] = dead_norm_filt_tot_int
                        #dead_norm_filt_avg_int = dead_norm_filt_tot_int / dead_area
                        #dead_norm_filt_tot_ints[fcn] = dead_norm_filt_avg_int
                    else:
                        norm_filt_dead = None
                        dead_norm_filt_tot_ints[fcn] = 0

                    #cv2.imwrite(os.path.join('E:/PyInstallerTests/VID871', f'{base_image_name[:-4]}_{fcn}_norm_alive.png'),norm_filt_alive)
                    #cv2.imwrite(os.path.join('E:/PyInstallerTests/VID871', f'{base_image_name[:-4]}_{fcn}_norm_dead.png'),norm_filt_dead)

                    normalized_filtered_conf = normalized_img_uint8.copy()
                    normalized_filtered_conf[conf_uint] = 0


                    if os.path.isdir(full_path_denoised):
                        
                        try:
                            if os.path.isfile(os.path.join(full_path_denoised, base_image_name)):
                                org_img = cv2.imread(os.path.join(full_path_denoised, base_image_name))
                    
                                if len(org_img.shape) == 2:
                                    img_gray = org_img
                                else:
                                    img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
                                    
                            else:
                                org_img = np.zeros((height,width), np.uint8)
                                img_gray = org_img

                    
                    
                            if alive_area > 0:
                                denoised_filt_alive = img_gray.copy()
                                denoised_filt_alive[alive_uint <= 0] = 0
                                alive_denoised_filt_tot_int = np.sum(denoised_filt_alive)
                                alive_denoised_filt_tot_ints[fcn] = alive_denoised_filt_tot_int
                                #alive_denoised_filt_avg_int = alive_denoised_filt_tot_int / alive_area
                                #alive_denoised_filt_tot_ints[fcn] = alive_denoised_filt_avg_int
                            else:
                                denoised_filt_alive = None
                                alive_denoised_filt_tot_ints[fcn] = 0
                    
                            if dead_area > 0:
                                denoised_filt_dead = img_gray.copy()
                                denoised_filt_dead[dead_uint == 0] = 0
                                dead_denoised_filt_tot_int = np.sum(denoised_filt_dead)
                                dead_denoised_filt_tot_ints[fcn] = dead_denoised_filt_tot_int
                                dead_denoised_filt_avg_int = dead_denoised_filt_tot_int / dead_area
                                #dead_denoised_filt_tot_ints[fcn] = dead_denoised_filt_avg_int

                                #if 'VID871_C10_1' in base_image_name and fcn == 'nir':
                                #    print(f'{base_image_name}: {dead_denoised_filt_tot_int}          avg: {dead_denoised_filt_avg_int}')



                            else:
                                denoised_filt_dead = None
                                dead_denoised_filt_tot_ints[fcn] = 0

                            #cv2.imwrite(os.path.join('E:/PyInstallerTests/VID871', f'{base_image_name[:-4]}_{fcn}_denoised_alive.png'),denoised_filt_alive)
                            #cv2.imwrite(os.path.join('E:/PyInstallerTests/VID871', f'{base_image_name[:-4]}_{fcn}_denoised_dead.png'),denoised_filt_dead)
                    
                            denoised_filtered_conf = img_gray.copy()
                            denoised_filtered_conf[conf_uint] = 0
                    
                            fcci = flourescence_channel_image(fcn, org_img, img_gray, 
                                                              normalized_img_uint8, norm_filt_alive, norm_filt_dead, normalized_filtered_conf, 
                                                              img_gray, denoised_filt_alive, denoised_filt_dead, denoised_filtered_conf)
                    
                            flourescence_channels.append(fcci)
                    
                        except Exception as e:
                            print(e)

                    else:

                        fcci = flourescence_channel_image(fcn, org_img, img_gray, 
                                                          normalized_img_uint8, norm_filt_alive, norm_filt_dead, normalized_filtered_conf,
                                                          None, None, None, None)

                    flourescence_channels.append(fcci)


                except Exception as e:
                    print(e)



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
            'Timestamp' : image_name_parts[3],
            'total cell count' : alive_count+dead_count,
            'alive cell count' : alive_count,
            'dead cell count' : dead_count,
            'confluence' : total_conf,
            'confluence alive' : alive_conf,
            'confluence dead' : dead_conf,
            'avg_alive_area' : avg_alive_area,
            'avg_dead_area' : avg_dead_area,
            }


        for fcn, intensity in alive_norm_filt_tot_ints.items():
            result_dict[f'alive_{fcn}_norm_intensity'] = intensity
            result_dict[f'mean_alive_{fcn}_norm_intensity'] = intensity / alive_count

        for fcn, intensity in alive_denoised_filt_tot_ints.items():
            result_dict[f'alive_{fcn}_denoised_intensity'] = intensity
            result_dict[f'mean_alive_{fcn}_denoised_intensity'] = intensity / alive_count

        for fcn, intensity in dead_norm_filt_tot_ints.items():
            result_dict[f'dead_{fcn}_norm_intensity'] = intensity
            result_dict[f'mean_dead_{fcn}_norm_intensity'] = intensity / dead_count

        for fcn, intensity in dead_denoised_filt_tot_ints.items():
            #if fcn == 'nir':
            #    print(f'nir intensity: {intensity}')

            result_dict[f'dead_{fcn}_denoised_intensity'] = intensity
            result_dict[f'mean_dead_{fcn}_denoised_intensity'] = intensity / dead_count


        cells_list = []

        for ac in analysed_cells_final:
            cell_dict = {
                'name': base_image_name[:-4],
                'x': ac.x,
                'y': ac.y,
                'state': ac.state
            }
            
            if flourescence_channels:
                for fcn, int_values in ac.avg_flourrescence_intesities.items():

                    cell_dict[f'{fcn}_org'] = int_values.org
                    cell_dict[f'{fcn}_norm'] = int_values.normalized
                    cell_dict[f'{fcn}_denoised'] = int_values.denoised
            
            cells_list.append(cell_dict)
        
        cells_df = pd.DataFrame(cells_list)

        


        return (result_dict, cells_df)



   









def on_magnification_dropdown_change(event_value):

    config['DEFAULT']['magnification'] = event_value

    with open('config.ini', 'w') as configfile:
        config.write(configfile)    

def on_resource_dropdown_change(event_value):

    config['DEFAULT']['resource_allocation'] = event_value
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)                     



def open_settings():
    print("Settings")
    config.read('config.ini')

    new_window = tk.Toplevel()
    new_window.title("Settings")
    new_window.geometry("550x570")
    new_window.lift()
    new_window.attributes('-topmost', True)

    

    # Frame for model selection
    denoiser_model_frame = tk.Frame(new_window)
    denoiser_model_frame.grid(row=0, column=0, sticky="nsew")

    modelname_analyser = str(config['DEFAULT']['modelname_analyser'])
    
    analyser_model_path_var = tk.StringVar(new_window, value=modelname_analyser)

    def open_analyser_model_dialog():
        new_window.attributes('-topmost', False)
        new_model_path_analyser = filedialog.askopenfilename(filetypes=[("TFlite Models", "*.tflite")])
        new_window.attributes('-topmost', True)

        if new_model_path_analyser:
            (config['DEFAULT']['model_path_analyser']) = new_model_path_analyser
            new_modelname_analyser_ext = os.path.basename(new_model_path_analyser)
            new_modelname_analyser, _ = os.path.splitext(new_modelname_analyser_ext)
            config['DEFAULT']['modelname_analyser'] = new_modelname_analyser
            analyser_model_path_var.set(new_modelname_analyser)

            tf.keras.backend.clear_session()
            gc.collect()

    def on_save_image_with_overlay():
        config['DEFAULT']['save_image_with_overlay'] = str(checkbox_overlay.get())

    def on_csv_decimal_dropdown_change(event_value):
        config['DEFAULT']['csv_decimal'] = str(event_value)

    analyser_model_frame = tk.Frame(new_window)
    analyser_model_frame.grid(row=1, column=0, sticky="nsew")

    select_model_button = ctk.CTkButton(analyser_model_frame, text="Select Analyser Model...", command=open_analyser_model_dialog)
    select_model_button.pack(side="left", pady=10, padx=5)
    
    model_analyser_label = ctk.CTkLabel(analyser_model_frame, textvariable=analyser_model_path_var)
    model_analyser_label.pack(side="left", pady=10, padx=5)

    # Frame for checkboxes 
    checkbox_frame = tk.Frame(new_window)
    checkbox_frame.grid(row=2, column=0, sticky="nsew")
    
    save_image_with_overlay_value = config['DEFAULT'].getboolean('save_image_with_overlay')  
    checkbox_overlay = ctk.CTkCheckBox(checkbox_frame, text="Save image with overlay?", command=on_save_image_with_overlay)
    checkbox_overlay.pack(pady=(10,50), padx=5, anchor='w')
    checkbox_overlay.select() if save_image_with_overlay_value else checkbox_overlay.deselect() 


    csv_decimal_frame = tk.Frame(new_window)
    csv_decimal_frame.grid(row=3, column=0, sticky="nsew")
    csv_decimal_options = [',', '.'] 
    csv_decimal_label = ctk.CTkLabel(csv_decimal_frame, text="Decimal Separator")  
    csv_decimal_label.pack()

    csv_decimal_dropdown = ctk.CTkComboBox(csv_decimal_frame, values=csv_decimal_options, command=on_csv_decimal_dropdown_change)
    #csv_decimal_dropdown.set(selected_value)
    csv_decimal_dropdown.pack()





    save_close_frame = tk.Frame(new_window)
    save_close_frame.grid(row=4, column=0, sticky="nsew")

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
    root.geometry('400x370')

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

    def on_use_denoiser():
        config['DEFAULT']['use_denoiser'] = str(checkbox_denoiser.get())
        with open('config.ini', 'w') as configfile:
            config.write(configfile)


    # Settings
    settings_menu = Menu(menubar, tearoff=0)
    settings_menu.add_command(label="Settings", command=open_settings)
    settings_menu.add_command(label="Quick Start", command=open_manual)
    settings_menu.add_command(label="Manual", command=open_manual)
    settings_menu.add_command(label="GitHub", command=open_github)
    #settings_menu.add_command(label="Citeation", command=open_settings)
    menubar.add_cascade(label="Menu", menu=settings_menu)

    root.config(menu=menubar)

    btn_open = ctk.CTkButton(root, text="Select Phase Image Directory", command=CA.on_select_bf_dir_clicked)
    btn_open.configure(width=220)
    btn_open.pack(pady=10)

    magnification_options = ["4x", "10x", "20x"] 
    magnification_label = ctk.CTkLabel(root, text="Magnification")  
    magnification_label.pack()

    magnification_dropdown = ctk.CTkComboBox(master=root, values=magnification_options, command=on_magnification_dropdown_change)
    magnification_dropdown.set(magnification)
    magnification_dropdown.pack()



    use_denoiser = config['DEFAULT'].getboolean('use_denoiser')  
    checkbox_denoiser = ctk.CTkCheckBox(master=root, text="Denoise Flourescence Images?", command=on_use_denoiser)
    checkbox_denoiser.pack(pady=(30,30), padx=5)
    checkbox_denoiser.select() if use_denoiser else checkbox_denoiser.deselect() 



    # Resource Selection Dropdown 
    selected_value = config['DEFAULT']['resource_allocation']
    resource_options = ["25", "50", "75", "100"] 
    resource_label = ctk.CTkLabel(root, text="CPU Resource Allocation (%)")  
    resource_label.pack()

    resource_dropdown = ctk.CTkComboBox(master=root, values=resource_options, command=on_resource_dropdown_change)
    resource_dropdown.set(selected_value)
    resource_dropdown.pack()

    # Buttons 
    

    btn_analyse = ctk.CTkButton(root, text="Analyse", command=CA.on_analyse_clicked)
    btn_analyse.configure(width=220)
    btn_analyse.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':  # Main guard
    multiprocessing.freeze_support()  
    main() 
