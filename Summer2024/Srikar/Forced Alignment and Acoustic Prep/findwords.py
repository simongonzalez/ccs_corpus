from praatio import tgio

def find_words(data=None, seg_number=None, tier_label=None, word_tier=None):

    segment_tier = data.tierDict[tier_label]
    word_tier = data.tierDict[word_tier]
    
    # Calculate final begin and end times for the segment
    fin_begin = segment_tier.entryList[seg_number][0]
    fin_end = segment_tier.entryList[seg_number][1]
    
    # Find indices of words that start before or at finBegin
    all_inds = [i for i, entry in enumerate(word_tier.entryList) if entry[0] <= fin_begin]
    
    # Get the index of the last occurrence
    unique_index = all_inds[-1]
    
    # Retrieve word properties
    word_label = word_tier.entryList[unique_index][2]
    word_onset = word_tier.entryList[unique_index][0]
    word_offset = word_tier.entryList[unique_index][1]
    word_mid = word_onset + ((word_offset - word_onset) / 2)
    word_dur = word_offset - word_onset
    
    # Return results as a list
    return [word_label, word_onset, word_offset, word_mid, word_dur, unique_index]
