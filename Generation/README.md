# Setup

Create a new conda environment using environment.yml.

# Creating a song

## Synthetization
First the melody is generated using synthesizer.ipynb
Train a model or use our pre-trained model to generate notes using the last few cells of the notebook.

## Rhythmization
Execute rhythmizer.py, as infile argument you set the .mid file containing the synthesized notes, as outfile argument you set 
how your output should be named, but without a file ending, as both a .mid and a .wav file will be created.

## Harmonization
The last step is executing the harmonizer.ipynb notebook and setting your rhythmized file name
and desired output name in the second code cell.
After training the evolutionary algorithm long enough, it will produce your final song!
