# Setup

Create a new conda environment using environment.yml.

# Creating a song

## Synthetization
First midi pitches are generated using music_generation_rnn.ipynb
Train a model or use our pre-trained model to generate pitches using the last few cells of the notebook.

## Rhythmization
Execute rhythmizer.py, as infile argument you set the .mid file containing the pitches, as outfile argument you set 
how your output should be named, but without a file ending, as both a .mid and a .wav file will be created.

## Harmonization
The last step is executing the evolutionary_pipeline_hyperparam.ipynb notebook and setting your rhythmized file name
and desired output name in the second code cell.
After training the evolutionary algorithm long enough, it will produce your final song!
