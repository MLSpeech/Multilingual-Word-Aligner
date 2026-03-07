from praatio import textgrid
import numpy as np
import matplotlib.pyplot as plt
import librosa
import matplotlib.gridspec as gridspec
from pathlib import Path
import os 
import seaborn as sns
import pandas as pd
from pathlib import Path

def dp_results_textgrid(textgrid_name, prob, predictions, dynamic_prediction_intervals, output_folder, units=10):
    """create text file from list of classes
    Args:
        textgrid_name (str): name of the file
        labels (list[int]): = [0,0,0,0,1,0,1,...]
        predictions (list[int]): = [0,0,0,0,1,0,1,...]
    """
    try:
        text_grids_dir = output_folder
        os.makedirs(text_grids_dir, exist_ok=True)
        
        times = np.array(list(range(0,len(prob)+1)))*units
        # Create intervals by pairing the times and labels
        prediction_intervals = [(str(times[i]), str(times[i+1]), str(predictions[i])) for i in range(len(predictions))]
        prob_intervals = [(str(times[i]), str(times[i+1]), str(prob[i])) for i in range(len(prob))]
        dynamic_prediction_intervals = [(str(times[i]), str(times[i+1]), str(dynamic_prediction_intervals[i])) for i in range(len(dynamic_prediction_intervals))]

        # Create IntervalTiers dynamically
        predictions_tier = textgrid.IntervalTier("Predictions", prediction_intervals, minT=min(times), maxT=max(times))
        probs_tier = textgrid.IntervalTier("Prob", prob_intervals, minT=min(times), maxT=max(times))
        dynamic_predictions_tier = textgrid.IntervalTier("Dynamic_Predictions", dynamic_prediction_intervals, minT=min(times), maxT=max(times))

        # Create the TextGrid
        tg = textgrid.Textgrid()
        tg.addTier(predictions_tier)
        tg.addTier(probs_tier)
        tg.addTier(dynamic_predictions_tier)

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


def extract_textgrid_results_to_image_dp(textgrid_path, wav_file):
    file_name = Path(textgrid_path).stem
    # Load the TextGrid file
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    # Get tiers from the TextGrid
    predictions_tier = tg.getTier('Predictions')  # Adjust based on your tier name
    probs_tier = tg.getTier('Prob')
    dynamic_predictions_tier = tg.getTier('Dynamic_Predictions')
    
    y, sr = librosa.load(wav_file, sr=16000)
    time = np.arange(0, len(y)) * (1/sr) * 1000  # Time in milliseconds

    # Extract intervals from the tier (start time, end time, label)
    p_intervals = predictions_tier.entries
    probs_intervals = probs_tier.entries
    dp_intervals = dynamic_predictions_tier.entries

    ### First Graph: y and probs_intervals over time with labels and predictions as vertical lines
    fig1 = plt.figure(figsize=(20, 15))
    ax1 = fig1.add_subplot(211)  # First subplot
    ax2 = fig1.add_subplot(212)  # Second subplot

    ax1.plot(time, y, color='gray')
    ax1.set_title(f"{file_name}_Waveform")
    ax1.set_ylabel("Amplitude")

    for start, end, prob in probs_intervals:
        ax2.plot([start, end], [float(prob), float(prob)], color='orange', lw=6)  # Use actual probability value
    ax2.set_title("Probabilities of Original Model")
    ax2.set_ylim(-0.1, 1.1) 
    ax2.set_yticks(np.arange(0, 1.1, 0.2)) 
    ax1.set_ylabel("Probability")

    # Plot vertical lines for labels and predictions, and set color to red for matching intervals
    for (start, end, label), (p_start, p_end, p_label) in zip(dp_intervals, p_intervals):
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
    ax1.text(1.01, 0.5, 'DP = Blue\nPredictions = Green\nMatching = Red', transform=ax1.transAxes, fontsize=12, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Align the x-limits of both subplots to make sure vertical lines are aligned
    ax1.set_xlim(0, tg.maxTimestamp)
    ax2.set_xlim(0, tg.maxTimestamp)

    # Save first figure
    fig1.savefig(f"{textgrid_path.replace('.TextGrid', '_graph1.png')}")
    plt.close(fig1)  # Close the figure to free memory

    ### Second Graph: y over time, labels, predictions, and dp_intervals over time
    # fig2 = plt.figure(figsize=(20, 15))
    # ax3 = fig2.add_subplot(411)  # First subplot
    # ax5 = fig2.add_subplot(413)  # Third subplot
    # ax6 = fig2.add_subplot(414)  # Fourth subplot

    # ax3.plot(time, y, color='gray')
    # ax3.set_title(f"{file_name}_Waveform")
    # ax3.set_ylabel("Amplitude")
    
    # for start, end, predictions in p_intervals:
    #     if float(predictions)!=0.0:
    #         ax5.plot([start, end], [1, 1], color='green', lw=6)
    #         ax5.text((start + end) / 2, 1.1, predictions, ha='center', va='bottom',fontsize=6)
    # ax5.set_title("Original Model Predictions")
    # ax5.set_ylim(0.8, 1.2)
    # ax5.set_yticks([])

    # for start, end, dp_prediction in dp_intervals:
    #     if float(dp_prediction)!=0.0:
    #         ax6.plot([start, end], [1, 1], color='green', lw=6)
    #         ax6.text((start + end) / 2, 1.1, dp_prediction, ha='center', va='bottom',fontsize=6)
    # ax6.set_title("DP Predictions")
    # ax6.set_ylim(0.8, 1.2)
    # ax6.set_yticks([])
    
    # # Align the x-limits of all subplots to make sure vertical lines are aligned
    # ax3.set_xlim(0, tg.maxTimestamp)
    # ax5.set_xlim(0, tg.maxTimestamp)
    # ax6.set_xlim(0, tg.maxTimestamp)

    # # Save second figure
    # fig2.savefig(f"{textgrid_path.replace('.TextGrid', '_graph2.png')}")
    # plt.close(fig2)  # Close the figure to free memory


def extract_graphs(graphs_name,predictions, prob, dp_predictions, **configuration):
    if configuration['no_graph']:
        return
    output_folder = configuration['output_folder']
    wav_file = configuration['wav_file']
    predictions, prob, dp_predictions = predictions.numpy(), prob.numpy(), dp_predictions.numpy()
    textgrid_file_path = dp_results_textgrid(graphs_name, prob, predictions, dp_predictions, output_folder)
    extract_textgrid_results_to_image_dp(textgrid_path=textgrid_file_path, wav_file=wav_file) 

def save_results_to_csv(sentence, dp_predictions_times, **configuration):
    if configuration['no_csv']:
        return
    words_end_times = list(np.array(dp_predictions_times)/configuration['dp_times_in_second'])
    words_start_times = [0]+words_end_times[:-1]
    results_df = pd.DataFrame(list(zip(sentence.split(), words_start_times, words_end_times)), columns=['Word', 'Start_Time', 'End_Time'])
    results_csv_path = os.path.join(configuration['output_folder'], f"{Path(configuration['wav_file']).stem}.csv")
    results_df.to_csv(results_csv_path, index=False)


# def the_graph_of_dp():
#     ### Third Graph: Plot y over time and heatmap of dp
#     fig3 = plt.figure(figsize=(20, 15))

#     # Create a GridSpec layout with more space for ax8 (heatmap)
#     gs = fig3.add_gridspec(2, 1, height_ratios=[1, 3])  # Allocate more space for ax8

#     # First subplot (ax7): Plot y (Waveform) over time
#     ax7 = fig3.add_subplot(gs[0])  # First row
#     ax7.plot(time, y, color='gray')
#     ax7.set_title(f"{file_name}_Waveform")
#     ax7.set_ylabel("Amplitude")

#     # Second subplot (ax8): Plot the heatmap of dp
#     ax8 = fig3.add_subplot(gs[1])  # Second row (more space)
#     # Process dp tensor: remove the first row only, then transpose
#     dp_processed = dp[1:, :].numpy()  # Convert to NumPy and remove the first row
#     dp_processed = dp_processed.T  # Transpose the matrix
#     print("dp_processed shape: ", dp_processed.shape)

#     # Duplicate each column 10 times
#     #dp_processed_repeated = np.tile(dp_processed, (1, 10))  # Repeat each column 10 times
#     dp_processed_repeated = np.repeat(dp_processed, 10, axis=1)
#     dp_processed_repeated = normalize_by_row_max(dp_processed_repeated)


#     # Plot the heatmap of the processed dp
#     im = ax8.imshow(dp_processed_repeated, cmap='viridis', aspect='auto')
#     fig3.colorbar(im, ax=ax8, orientation='vertical')

#     # Overlay labels as vertical lines
#     for start, end, label in l_intervals:
#         if float(label) != 0.0:
#             ax7.axvline(x=start, color='pink', lw=2)
#             ax8.axvline(x=start, color='pink', lw=2)

#     # Add horizontal lines in ax8 between rows in dp
#     num_rows = dp_processed_repeated.shape[0]  # Get the number of rows after processing
#     for row in range(1, num_rows):  # Start from row 1 to avoid placing a line at the top
#         ax8.axhline(y=row - 0.5, color='black', lw=1)  # -0.5 places the line between rows

#     # Adjust the x-limits of both subplots
#     ax8.set_xlabel("Time (ms)")
#     ax7.set_xlim(0, tg.maxTimestamp)
#     ax8.set_xlim(0, tg.maxTimestamp)

#     # Save third figure
#     fig3.savefig(f"{textgrid_path.replace('.TextGrid', '_graph3.png')}")
#     plt.close(fig3)  # Close the figure to free memory

#     print(f"Figures saved for {textgrid_path}")
    