import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Traverse upwards until the directory name matches the given project name
project_dir = current_dir
while os.path.basename(project_dir) != "MWA":
    project_dir = os.path.dirname(project_dir)
    if project_dir == '/':  # Breaks if we reach the root of the filesystem
        raise Exception(f"Project directory 'MWA' not found. Ensure this script is within the project directory.")

# Append the project directory to sys.path
sys.path.append(project_dir)

from praatio import textgrid
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import seaborn as sns
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D


def dp_results_textgrid(textgrid_name, labels, prob, predictions, dynamic_prediction_intervals, mms_preds_10ms_vector, unsupseg_preds_vector, unsupseg_probabilities, text_grids_dir, units=10):
    """create text file from list of classes
    Args:
        textgrid_name (str): name of the file
        labels (list[int]): = [0,0,0,0,1,0,1,...]
        predictions (list[int]): = [0,0,0,0,1,0,1,...]
    """
    try:
        os.makedirs(text_grids_dir, exist_ok=True)
        
        times = np.array(list(range(0,len(labels)+1)))*units
        # Create intervals by pairing the times and labels
        label_intervals = [(str(times[i]), str(times[i+1]), str(labels[i])) for i in range(len(labels))]
        prediction_intervals = [(str(times[i]), str(times[i+1]), str(predictions[i])) for i in range(len(predictions))]
        prob_intervals = [(str(times[i]), str(times[i+1]), str(prob[i])) for i in range(len(prob))]
        dynamic_prediction_intervals = [(str(times[i]), str(times[i+1]), str(dynamic_prediction_intervals[i])) for i in range(len(dynamic_prediction_intervals))]
        mms_prediction_intervals = [(str(times[i]), str(times[i+1]), str(mms_preds_10ms_vector[i])) for i in range(len(mms_preds_10ms_vector))]
        unsupseg_prediction_intervals = [(str(times[i]), str(times[i+1]), str(unsupseg_preds_vector[i])) for i in range(len(unsupseg_preds_vector))]
        unsupseg_prob_intervals = [(str(times[i]), str(times[i+1]), str(unsupseg_probabilities[i])) for i in range(len(unsupseg_probabilities))]

        # Create IntervalTiers dynamically
        labels_tier = textgrid.IntervalTier("Labels", label_intervals, minT=min(times), maxT=max(times))
        predictions_tier = textgrid.IntervalTier("Predictions", prediction_intervals, minT=min(times), maxT=max(times))
        probs_tier = textgrid.IntervalTier("Prob", prob_intervals, minT=min(times), maxT=max(times))
        dynamic_predictions_tier = textgrid.IntervalTier("Dynamic_Predictions", dynamic_prediction_intervals, minT=min(times), maxT=max(times))
        mms_prediction_tier = textgrid.IntervalTier("MMS_Predictions", mms_prediction_intervals, minT=min(times), maxT=max(times))
        unsupseg_predictions_tier = textgrid.IntervalTier("UnSupSeg_Predictions", unsupseg_prediction_intervals, minT=min(times), maxT=max(times))
        unsupseg_probs_tier = textgrid.IntervalTier("UnSupSeg_probabilities", unsupseg_prob_intervals, minT=min(times), maxT=max(times))

        # Create the TextGrid
        tg = textgrid.Textgrid()
        tg.addTier(labels_tier)
        tg.addTier(predictions_tier)
        tg.addTier(probs_tier)
        tg.addTier(dynamic_predictions_tier)
        tg.addTier(mms_prediction_tier)
        tg.addTier(unsupseg_predictions_tier)
        tg.addTier(unsupseg_probs_tier)

        # Save the TextGrid file
        textgrid_file_path = os.path.join(text_grids_dir,f"{textgrid_name}.TextGrid")
        tg.save(textgrid_file_path, format="long_textgrid", includeBlankSpaces=True)
        return textgrid_file_path
    except Exception as e:
        print(f"unable to create textgrid for {textgrid_name}:\n{e}")
        return None



def normalize_by_row_max(arr):
    # Find the maximum value in each row
    row_maxs = np.max(arr, axis=1, keepdims=True)
    # Normalize each row by dividing it by its maximum value
    normalized_arr = arr / row_maxs
    return normalized_arr


def extract_textgrid_results_to_image_dp(textgrid_path, wav_file, embedding, dp):
    file_name = Path(textgrid_path).stem
    # Load the TextGrid file
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    # Get tiers from the TextGrid
    labels_tier = tg.getTier('Labels')  # Adjust based on your tier name
    predictions_tier = tg.getTier('Predictions')  # Adjust based on your tier name
    probs_tier = tg.getTier('Prob')
    dynamic_predictions_tier = tg.getTier('Dynamic_Predictions')
    mms_prediction_tier = tg.getTier('MMS_Predictions')
    unsupseg_predictions_tier = tg.getTier('UnSupSeg_Predictions')
    unsupseg_probs_tier = tg.getTier('UnSupSeg_probabilities')
    
    y, sr = librosa.load(wav_file, sr=16000)
    time = np.arange(0, len(y)) * (1/sr) * 1000  # Time in milliseconds

    # Extract intervals from the tier (start time, end time, label)
    l_intervals = labels_tier.entries
    p_intervals = predictions_tier.entries
    probs_intervals = probs_tier.entries
    dp_intervals = dynamic_predictions_tier.entries
    mms_intervals = mms_prediction_tier.entries
    unsupseg_pred_intervals = unsupseg_predictions_tier.entries
    unsupseg_prob_intervals = unsupseg_probs_tier.entries

    ### First Graph: y and probs_intervals over time with labels and predictions as vertical lines
    fig1 = plt.figure(figsize=(20, 15))
    ax1 = fig1.add_subplot(211)  # First subplot
    ax2 = fig1.add_subplot(212)  # Second subplot

    ax1.plot(time, y, color='gray')
    ax1.set_title(f"Audio Waveform - labels and sequence model predictions")
    ax1.set_ylabel("Amplitude")

    for start, end, prob in probs_intervals:
        ax2.plot([start, end], [float(prob), float(prob)], color='orange', lw=6)  # Use actual probability value
    ax2.set_title("Probabilities of Original Model")
    ax2.set_ylim(-0.1, 1.1) 
    ax2.set_yticks(np.arange(0, 1.1, 0.2)) 
    ax1.set_ylabel("Probability")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, p_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax1.axvline(x=start, color='red', lw=2)
                ax2.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax1.axvline(x=start, color='blue', lw=2)
                ax2.axvline(x=start, color='blue', lw=2)
            else:
                ax1.axvline(x=p_start, color='green', lw=2)
                ax2.axvline(x=p_start, color='green', lw=2)        

    # Add a textual annotation on the side of the plot explaining the colors
    ax1.text(1.01, 0.5, 'Labels = Blue\nPredictions = Green\nMatching = Red', transform=ax1.transAxes, fontsize=12, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Align the x-limits of both subplots to make sure vertical lines are aligned
    ax1.set_xlim(0, tg.maxTimestamp)
    ax2.set_xlim(0, tg.maxTimestamp)

    # Save first figure
    fig1.savefig(f"{textgrid_path.replace('.TextGrid', '_graph1.png')}")
    plt.close(fig1)  # Close the figure to free memory

    ##########################

    ### Second Graph: y over time, labels, predictions, and dp_intervals over time
    fig2 = plt.figure(figsize=(20, 15))
    ax3 = fig2.add_subplot(411)  # First subplot
    ax4 = fig2.add_subplot(412)  # Second subplot
    ax5 = fig2.add_subplot(413)  # Third subplot
    ax6 = fig2.add_subplot(414)  # Fourth subplot

    ax3.plot(time, y, color='gray')
    ax3.set_title(f"{file_name}_Waveform")
    ax3.set_ylabel("Amplitude")
    
    
    for start, end, label in l_intervals:
        if float(label)!=0.0:
            ax4.plot([start, end], [1, 1], color='blue', lw=6)
            ax4.text((start + end) / 2, 1.1, label, ha='center', va='bottom',fontsize=6)
    ax4.set_title("Labels")
    ax4.set_ylim(0.8, 1.2)
    ax4.set_yticks([])

    
    for start, end, predictions in p_intervals:
        if float(predictions)!=0.0:
            ax5.plot([start, end], [1, 1], color='green', lw=6)
            ax5.text((start + end) / 2, 1.1, predictions, ha='center', va='bottom',fontsize=6)
    ax5.set_title("Original Model Predictions")
    ax5.set_ylim(0.8, 1.2)
    ax5.set_yticks([])

    for start, end, dp_prediction in dp_intervals:
        if float(dp_prediction)!=0.0:
            ax6.plot([start, end], [1, 1], color='green', lw=6)
            ax6.text((start + end) / 2, 1.1, dp_prediction, ha='center', va='bottom',fontsize=6)
    ax6.set_title("DP Predictions")
    ax6.set_ylim(0.8, 1.2)
    ax6.set_yticks([])
    
    # Align the x-limits of all subplots to make sure vertical lines are aligned
    ax3.set_xlim(0, tg.maxTimestamp)
    ax4.set_xlim(0, tg.maxTimestamp)
    ax5.set_xlim(0, tg.maxTimestamp)
    ax6.set_xlim(0, tg.maxTimestamp)

    # Save second figure
    fig2.savefig(f"{textgrid_path.replace('.TextGrid', '_graph2.png')}")
    plt.close(fig2)  # Close the figure to free memory

    ##########################

    ### Third Graph: Plot y over time and heatmap of dp
    fig3 = plt.figure(figsize=(20, 15))

    # Create a GridSpec layout with more space for ax8 (heatmap)
    gs = fig3.add_gridspec(2, 1, height_ratios=[1, 3])  # Allocate more space for ax8

    # First subplot (ax7): Plot y (Waveform) over time
    ax7 = fig3.add_subplot(gs[0])  # First row
    ax7.plot(time, y, color='gray')
    ax7.set_title(f"{file_name}_Waveform")
    ax7.set_ylabel("Amplitude")

    # Second subplot (ax8): Plot the heatmap of dp
    ax8 = fig3.add_subplot(gs[1])  # Second row (more space)
    # Process dp tensor: remove the first row only, then transpose
    dp_processed = dp[1:, :].numpy()  # Convert to NumPy and remove the first row
    dp_processed = dp_processed.T  # Transpose the matrix
    print("dp_processed shape: ", dp_processed.shape)

    # Duplicate each column 10 times
    #dp_processed_repeated = np.tile(dp_processed, (1, 10))  # Repeat each column 10 times
    dp_processed_repeated = np.repeat(dp_processed, 10, axis=1)
    dp_processed_repeated = normalize_by_row_max(dp_processed_repeated)


    # Plot the heatmap of the processed dp
    im = ax8.imshow(dp_processed_repeated, cmap='viridis', aspect='auto')
    fig3.colorbar(im, ax=ax8, orientation='vertical')

    # Overlay labels as vertical lines
    for start, end, label in l_intervals:
        if float(label) != 0.0:
            ax7.axvline(x=start, color='pink', lw=2)
            ax8.axvline(x=start, color='pink', lw=2)

    # Add horizontal lines in ax8 between rows in dp
    num_rows = dp_processed_repeated.shape[0]  # Get the number of rows after processing
    for row in range(1, num_rows):  # Start from row 1 to avoid placing a line at the top
        ax8.axhline(y=row - 0.5, color='black', lw=1)  # -0.5 places the line between rows

    # Adjust the x-limits of both subplots
    ax8.set_xlabel("Time (ms)")
    ax7.set_xlim(0, tg.maxTimestamp)
    ax8.set_xlim(0, tg.maxTimestamp)

    # Save third figure
    fig3.savefig(f"{textgrid_path.replace('.TextGrid', '_graph3.png')}")
    plt.close(fig3)  # Close the figure to free memory


    ##########################

    fig5 = plt.figure(figsize=(20, 20))
    ax9 = fig5.add_subplot(311)  # First subplot
    ax10 = fig5.add_subplot(312)  # Second subplot
    ax11 = fig5.add_subplot(313)  # Second subplot

    ax9.plot(time, y, color='gray')
    ax9.set_title(f"Audio Waveform - labels and sequence model predictions")
    ax9.set_ylabel("Amplitude")

    ax11.plot(time, y, color='gray')
    ax11.set_title(f"Audio Waveform - labels and DP predictions")
    ax11.set_ylabel("Amplitude")

    for start, end, prob in probs_intervals:
        ax10.plot([start, end], [float(prob), float(prob)], color='orange', lw=6)  # Use actual probability value
    ax10.set_title("Probabilities of Original Model")
    ax10.set_ylim(-0.1, 1.1) 
    ax10.set_yticks(np.arange(0, 1.1, 0.2)) 
    ax10.set_ylabel("Probability")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, p_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax9.axvline(x=start, color='red', lw=2)
                ax10.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax9.axvline(x=start, color='blue', lw=2)
                ax10.axvline(x=start, color='blue', lw=2)
            else:
                ax9.axvline(x=p_start, color='green', lw=2)
                ax10.axvline(x=p_start, color='green', lw=2)    

    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, dp_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax11.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax11.axvline(x=start, color='blue', lw=2)
            else:
                ax11.axvline(x=p_start, color='green', lw=2)


    # Add a textual annotation on the side of the plot explaining the colors
    ax9.text(1.01, 0.5, 'Labels = Blue\nPredictions = Green\nMatching = Red', transform=ax9.transAxes, fontsize=12, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    ax11.text(1.01, 0.5, 'Labels = Blue\nPredictions = Green\nMatching = Red', transform=ax11.transAxes, fontsize=12, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Align the x-limits of both subplots to make sure vertical lines are aligned
    ax9.set_xlim(0, tg.maxTimestamp)
    ax10.set_xlim(0, tg.maxTimestamp)
    ax11.set_xlim(0, tg.maxTimestamp)


    # Save first figure
    fig5.savefig(f"{textgrid_path.replace('.TextGrid', '_graph4.png')}")
    plt.close(fig5)  # Close the figure to free memory



    ##########################

    fig6 = plt.figure(figsize=(20, 20))
    ax12 = fig6.add_subplot(411)
    ax13 = fig6.add_subplot(412)
    ax14 = fig6.add_subplot(413)
    ax15 = fig6.add_subplot(414)

    plt.subplots_adjust(
        left=0.08,    # Reduced from default 0.125
        right=0.92,   # Reduced from default 0.9
        bottom=0.1,   # Small bottom margin for legend
        top=0.92,     # Reduced from default 0.88
        wspace=0.3,   # Horizontal space between subplots
        hspace=0.3    # Vertical space between subplots
    )


    # ax12 - plots the wav + the model DP predictions
    ax12.plot(time, y, color='gray')
    ax12.set_title(f"Audio Waveform - labels and MWA DP predictions")
    ax12.set_ylabel("Amplitude")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, dp_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax12.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax12.axvline(x=start, color='blue', lw=2)
            else:
                ax12.axvline(x=p_start, color='green', lw=2)

    # ax12.text(1.01, 0.5, 'Labels = Blue\nPredictions = Green\nMatching = Red', transform=ax12.transAxes, fontsize=12, verticalalignment='center',
    #          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # ax13 - Plots the conformer probabilities + predictios
    for start, end, prob in probs_intervals:
        ax13.plot([start, end], [float(prob), float(prob)], color='orange', lw=6)  # Use actual probability value
    ax13.set_title("Probabilities of Original Model")
    ax13.set_ylim(-0.1, 1.1) 
    ax13.set_yticks(np.arange(0, 1.1, 0.2)) 
    ax13.set_ylabel("Probability")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, p_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax13.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax13.axvline(x=start, color='blue', lw=2)
            else:
                ax13.axvline(x=p_start, color='green', lw=2)


    # ax14 - plots the wav + the MMS predictions
    ax14.plot(time, y, color='gray')
    ax14.set_title(f"Audio Waveform - labels and MMS predictions")
    ax14.set_ylabel("Amplitude")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, mms_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax14.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax14.axvline(x=start, color='blue', lw=2)
            else:
                ax14.axvline(x=p_start, color='green', lw=2)

    # ax14.text(1.01, 0.5, 'Labels = Blue\nPredictions = Green\nMatching = Red', transform=ax14.transAxes, fontsize=12, verticalalignment='center',
    #          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    

    # ax15 - Plots the UnSupSeg model probabilities + predictios
    # for start, end, prob in unsupseg_prob_intervals:
    #     ax15.plot([start, end], [float(prob), float(prob)], color='orange', lw=6)  # Use actual probability value
    # ax15.set_title("Probabilities of UnSupSeg model")
    # ax15.set_ylim(-0.1, 1.1) 
    # ax15.set_yticks(np.arange(0, 1.1, 0.2)) 
    # ax15.set_ylabel("Probability")

    # Assuming unsupseg_prob_intervals contains (start, end, prob) tuples
    starts = [x[0] for x in unsupseg_prob_intervals]
    ends = [x[1] for x in unsupseg_prob_intervals]
    probs = [float(x[2]) for x in unsupseg_prob_intervals]

    # Create midpoint x-values for smoother interpolation
    x_vals = np.array([(start + end)/2 for start, end in zip(starts, ends)])
    y_vals = np.array(probs)

    # Create interpolation function (cubic gives smooth curves)
    f = interp1d(x_vals, y_vals, kind='cubic', fill_value='extrapolate')

    # Generate smooth x-values
    x_smooth = np.linspace(min(starts), max(ends), 500)
    y_smooth = f(x_smooth)

    # Plot the smooth curve
    ax15.plot(x_smooth, y_smooth, color='orange', lw=2)

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, unsupseg_pred_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:  # If the intervals match, use red color
                ax15.axvline(x=start, color='red', lw=2)
            elif float(label) != 0.0:
                ax15.axvline(x=start, color='blue', lw=2)
            else:
                ax15.axvline(x=p_start, color='green', lw=2)



    

    # Your plotting code would go here...

    # Create larger legend patches with bigger font
    # legend_elements = [
    #     mpatches.Patch(color='blue', label='Labels'),
    #     mpatches.Patch(color='green', label='Predictions'),
    #     mpatches.Patch(color='red', label='Matching')
    # ]

    legend_elements = [
        Line2D([0], [0], color='blue', lw=6, label='Labels'),        # lw=6 makes thick line
        Line2D([0], [0], color='green', lw=6, label='Predictions'),  # lw=6 makes thick line
        Line2D([0], [0], color='red', lw=6, label='Matching')       # lw=6 makes thick line
    ]

    # Add large legend at the bottom
    legend = fig6.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),  # Negative value pushes it below axes
        fontsize=15,                  # Big font
        handlelength=2,               # Big color squares
        handleheight=2,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # [left, bottom, right, top]


    # Align the x-limits of both subplots to make sure vertical lines are aligned
    ax12.set_xlim(0, tg.maxTimestamp)
    ax13.set_xlim(0, tg.maxTimestamp)
    ax14.set_xlim(0, tg.maxTimestamp)
    ax15.set_xlim(0, tg.maxTimestamp)


    # Save first figure
    fig6.savefig(f"{textgrid_path.replace('.TextGrid', '_graph5.png')}")
    plt.close(fig6)  # Close the figure to free memory



    ##########


    fig7 = plt.figure(figsize=(20, 20))
    ax16 = fig7.add_subplot(411)
    ax17 = fig7.add_subplot(412)
    ax18 = fig7.add_subplot(413)
    ax19 = fig7.add_subplot(414)


    # ax16 - plots the wav + the model DP predictions
    ax16.plot(time, y, color='gray')
    ax16.set_title(f"Audio Waveform - labels and MWA predictions")
    ax16.set_ylabel("Amplitude")


    # Flags to track if we've added the labels to legend
    label_added = False
    pred_added = False

    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, dp_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:
                # Draw matching intervals with dashed lines
                ax16.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
                ax16.axvline(x=start, color='seagreen', linestyle='--', label='Prediction' if not pred_added else None, lw=2)
                label_added = True
                pred_added = True
            elif float(label) != 0.0:
                # Draw label line
                ax16.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
                label_added = True
            else:
                # Draw prediction line
                ax16.axvline(x=start, color='seagreen', linestyle='--', label='Prediction' if not pred_added else None, lw=2)
                pred_added = True

    ax16.legend(loc='upper right', fontsize=10)


    # ax17 - Plots the conformer probabilities + predictios
    label_added = False

    starts = [x[0] for x in probs_intervals]
    ends = [x[1] for x in probs_intervals]
    probs = [float(x[2]) for x in probs_intervals]

    # Create midpoint x-values for smoother interpolation
    x_vals = np.array([(start + end)/2 for start, end in zip(starts, ends)])
    y_vals = np.array(probs)

    # Create interpolation function (cubic gives smooth curves)
    f = interp1d(x_vals, y_vals, kind='cubic', fill_value='extrapolate')

    # Generate smooth x-values
    x_smooth = np.linspace(min(starts), max(ends), 500)
    y_smooth = f(x_smooth)

    # Plot the smooth curve
    ax17.plot(x_smooth, y_smooth, color='darkslateblue', lw=2)

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label) in l_intervals:
        if float(label) != 0.0:
            ax17.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
            label_added = True

    ax17.set_title("Probabilities of Sequence model - Conformer")
    ax17.legend(loc='upper right', fontsize=10)
    ax17.set_ylabel("Probability")

    # ax18 - plots the wav + the MMS predictions

    ax18.plot(time, y, color='gray')
    ax18.set_title(f"Audio Waveform - labels and MMS predictions")
    ax18.set_ylabel("Amplitude")

    # Flags to track if we've added the labels to legend
    label_added = False
    pred_added = False

    for (start, end, label), (p_start, p_end, p_label) in zip(l_intervals, mms_intervals):
        if float(label) != 0.0 or float(p_label) != 0.0:
            if start == p_start and end == p_end and float(p_label) != 0.0 and float(label) != 0:
                # Draw matching intervals with dashed lines
                ax18.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
                ax18.axvline(x=start, color='darkgreen', lw=2, linestyle='--', label='MMS Prediction' if not pred_added else None)
                label_added = True
                pred_added = True
            elif float(label) != 0.0:
                # Draw label line
                ax18.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
                label_added = True
            else:
                # Draw prediction line
                ax18.axvline(x=start, color='darkgreen', lw=2, linestyle='--', label='MMS Prediction' if not pred_added else None)
                pred_added = True

    ax18.legend(loc='upper right', fontsize=10)


    # ax 19 - UnSupSeg ploting
    # Assuming unsupseg_prob_intervals contains (start, end, prob) tuples
    label_added = False

    starts = [x[0] for x in unsupseg_prob_intervals]
    ends = [x[1] for x in unsupseg_prob_intervals]
    probs = [float(x[2]) for x in unsupseg_prob_intervals]

    # Create midpoint x-values for smoother interpolation
    x_vals = np.array([(start + end)/2 for start, end in zip(starts, ends)])
    y_vals = np.array(probs)

    # Create interpolation function (cubic gives smooth curves)
    f = interp1d(x_vals, y_vals, kind='cubic', fill_value='extrapolate')

    # Generate smooth x-values
    x_smooth = np.linspace(min(starts), max(ends), 500)
    y_smooth = f(x_smooth)

    # Plot the smooth curve
    ax19.plot(x_smooth, y_smooth, color='darkslateblue', lw=2)

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label) in l_intervals:
        if float(label) != 0.0:
            ax19.axvline(x=start, color='orangered', lw=2, linestyle='-', label='Label' if not label_added else None)
            label_added = True

    ax19.set_title("Probabilities of UnSupSeg model")
    ax19.legend(loc='upper right', fontsize=10)
    ax19.set_ylabel("Probability")



    # Align the x-limits of both subplots to make sure vertical lines are aligned
    ax16.set_xlim(0, tg.maxTimestamp)
    ax17.set_xlim(0, tg.maxTimestamp)
    ax18.set_xlim(0, tg.maxTimestamp)
    ax19.set_xlim(0, tg.maxTimestamp)


    # Save first figure
    fig7.savefig(f"{textgrid_path.replace('.TextGrid', '_graph6.png')}")
    plt.close(fig7)  # Close the figure to free memory

    print(f"Figures saved for {textgrid_path}")



def plot_heatmaps(path, dpss):
    tensor_keys = ['score_frame', 'mms_emission_score', 'score_word', 'distance_score']
    os.makedirs(path, exist_ok=True)
    
    for key in tensor_keys:
        plt.figure(figsize=(12, 6))
        tensor = dpss[key].T  # Transpose
        
        sns.heatmap(tensor, cmap='viridis', cbar=True, xticklabels=True, yticklabels=True)
        plt.xlabel('Frame')
        plt.ylabel('Word')
        plt.title(f'{key} Heatmap')

        # ===== KEY CHANGES =====
        num_frames = tensor.shape[1]
        step = 50  # Now showing every 50 frames (was 10)
        xticks = range(0, num_frames, step)
        
        # Set ticks horizontally (rotation=0) and center-aligned
        plt.xticks(
            xticks, 
            labels=[str(i) for i in xticks], 
            fontsize=10, 
            rotation=0,  # Horizontal labels (default is 90°)
            ha='center'  # Center-align labels under ticks
        )
        # ======================

        heatmap_filename = os.path.join(path, f'{key}_heatmap.png')
        plt.savefig(heatmap_filename, bbox_inches='tight', dpi=300)  # Added dpi for higher resolution
        plt.close()
        print(f"Saved {heatmap_filename}")

def plot_tensor_and_save(tensor, save_path):
    # Ensure the tensor is a 1D tensor (vector)
    tensor_np = tensor.cpu().numpy()

    # Create a figure and axis for the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the tensor values and connect the dots with a line
    plt.plot(tensor_np, marker='o', linestyle='-', color='b')

    # Add labels and a title
    plt.title("Tensor Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # Add grid for better visualization
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(save_path)

    # Close the plot to free memory (optional)
    plt.close()


def plot_emissions(array_2d, save_path, dpi=600, cmap='viridis'):
    """
    Transpose a 2D numpy array and plot as a high-res heatmap with custom y-axis text.
    """
    # Transpose the array
    transposed_array = np.transpose(array_2d)
    transposed_array = (transposed_array - transposed_array.min()) / (transposed_array.max() - transposed_array.min())
    
    # Create figure with tighter layout
    fig, ax = plt.subplots(figsize=(16, 5), dpi=dpi)
    fig.subplots_adjust(left=0.15)  # Reduced left margin
    
    # Heatmap with title
    heatmap = ax.imshow(transposed_array, 
                       aspect='auto', 
                       cmap=cmap,
                       interpolation='nearest')
    ax.set_title('MMS - Emissions', fontsize=12, pad=10)
    
    # Colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    # X-axis (0, 50, 100, 150...)
    x_max = transposed_array.shape[1]
    x_ticks = np.arange(0, x_max, 50)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)
    ax.set_xlabel('Frames (10 msec intervals)', fontsize=10)
    
    # Y-axis (0, 5, 10, 15...)
    y_max = transposed_array.shape[0]
    y_ticks = np.arange(0, y_max, 5)  # Every 5 units
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=8)
    ax.set_ylabel('Tokens - letters', fontsize=10)
    
    # Save
    if not save_path.endswith('.png'):
        save_path += '.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Saved final heatmap to {save_path}")