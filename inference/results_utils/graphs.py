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

def dp_results_textgrid(textgrid_name, prob, predictions, dynamic_prediction_intervals, units=10):
    """Build an in-memory TextGrid from frame-level predictions (no file I/O).

    Args:
        textgrid_name (str): name used for error messages
        prob (array): probability values per frame
        predictions (array): model prediction per frame
        dynamic_prediction_intervals (array): DP prediction per frame
        units (int): duration of each frame in ms
    Returns:
        Textgrid object, or None on failure
    """
    try:
        times = np.array(list(range(0, len(prob)+1))) * units
        prediction_intervals = [(str(times[i]), str(times[i+1]), str(predictions[i])) for i in range(len(predictions))]
        prob_intervals = [(str(times[i]), str(times[i+1]), str(prob[i])) for i in range(len(prob))]
        dynamic_prediction_intervals = [(str(times[i]), str(times[i+1]), str(dynamic_prediction_intervals[i])) for i in range(len(dynamic_prediction_intervals))]

        predictions_tier = textgrid.IntervalTier("Predictions", prediction_intervals, minT=min(times), maxT=max(times))
        probs_tier = textgrid.IntervalTier("Prob", prob_intervals, minT=min(times), maxT=max(times))
        dynamic_predictions_tier = textgrid.IntervalTier("Dynamic_Predictions", dynamic_prediction_intervals, minT=min(times), maxT=max(times))

        tg = textgrid.Textgrid()
        tg.addTier(predictions_tier)
        tg.addTier(probs_tier)
        tg.addTier(dynamic_predictions_tier)
        return tg
    except Exception as e:
        print(f"unable to create textgrid for {textgrid_name}:\n{e}")
        return None



def normalize_by_row_max(arr):
    # Find the maximum value in each row
    row_maxs = np.max(arr, axis=1, keepdims=True)
    # Normalize each row by dividing it by its maximum value
    normalized_arr = arr / row_maxs
    return normalized_arr


def extract_textgrid_results_to_image_dp(tg, file_name, wav_file, output_folder):
    y, sr = librosa.load(wav_file, sr=16000)
    time = np.arange(0, len(y)) * (1/sr) * 1000  # Time in milliseconds

    # Get tiers directly from the in-memory TextGrid object
    p_intervals = tg.getTier('Predictions').entries
    probs_intervals = tg.getTier('Prob').entries
    dp_intervals = tg.getTier('Dynamic_Predictions').entries

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
    fig1.savefig(os.path.join(output_folder, f"{file_name}_graph1.png"))
    plt.close(fig1)  # Close the figure to free memory


def extract_graphs(graphs_name, predictions, prob, dp_predictions, **configuration):
    if configuration['no_graph']:
        return
    output_folder = configuration['output_folder']
    wav_file = configuration['wav_file']
    predictions, prob, dp_predictions = predictions.numpy(), prob.numpy(), dp_predictions.numpy()
    tg = dp_results_textgrid(graphs_name, prob, predictions, dp_predictions)
    if tg is not None:
        extract_textgrid_results_to_image_dp(tg, graphs_name, wav_file, output_folder)


def save_results_to_csv(sentence, dp_predictions_times, **configuration):
    if configuration['no_csv']:
        return
    words_end_times = list(np.array(dp_predictions_times)/configuration['dp_times_in_second'])
    words_start_times = [0]+words_end_times[:-1]
    results_df = pd.DataFrame(list(zip(sentence.split(), words_start_times, words_end_times)), columns=['Word', 'Start_Time', 'End_Time'])
    results_csv_path = os.path.join(configuration['output_folder'], f"{Path(configuration['wav_file']).stem}.csv")
    results_df.to_csv(results_csv_path, index=False)


def save_results_to_textgrid(sentence, dp_predictions_times, **configuration):
    """Save word-level alignment as a TextGrid with second-based timestamps.

    Produces a single 'words' tier with one interval per word, plus a trailing
    empty interval from the last word boundary to the end of the audio file.
    """
    if configuration.get('no_csv'):
        return None

    wav_file = configuration['wav_file']
    output_folder = configuration['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    # Convert frame indices to seconds
    words_end_times = list(np.array(dp_predictions_times) / configuration['dp_times_in_second'])
    words_start_times = [0.0] + words_end_times[:-1]
    words = sentence.split()

    # Get total audio duration for the trailing empty interval
    y, sr = librosa.load(wav_file, sr=None)
    audio_duration = len(y) / sr

    # Build word intervals: (xmin, xmax, text)
    intervals = [
        (str(words_start_times[i]), str(words_end_times[i]), words[i])
        for i in range(len(words))
    ]

    # Append trailing empty interval up to the audio file's end
    if words_end_times[-1] < audio_duration:
        intervals.append((str(words_end_times[-1]), str(audio_duration), ""))

    words_tier = textgrid.IntervalTier("words", intervals, minT=0, maxT=audio_duration)

    tg = textgrid.Textgrid()
    tg.addTier(words_tier)

    file_stem = Path(wav_file).stem
    tg_path = os.path.join(output_folder, f"{file_stem}.TextGrid")
    tg.save(tg_path, format="long_textgrid", includeBlankSpaces=True)
    return tg_path