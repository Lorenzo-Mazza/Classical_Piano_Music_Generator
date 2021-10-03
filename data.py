from music21 import converter,note,chord
import os
import pandas as pd
import preprocessing
import numpy as np

def load_data(path='maestro-v2.0.0', batch_size=32):
    os.chdir(path)
    infos = pd.read_csv("maestro-v2.0.0.csv")
    composers = infos['canonical_composer'].unique().tolist()
    composers_midi = {}
    notes = []
    durations = []
    velocities = []
    counter=0
    for name, author in zip(infos['midi_filename'], infos['canonical_composer']):
        piece = converter.parse(name)
        if author not in composers_midi:
            composers_midi[author] = []
        composers_midi[author].append(piece)

        notes.extend(['START'] * batch_size)
        durations.extend([0] * batch_size)
        velocities.extend([0] * batch_size)
        for element in piece.flat:
            if isinstance(element, note.Note) or isinstance(element, note.Rest):
                if element.isRest:
                    notes.append(str(element.name))
                    durations.append(element.duration.quarterLength)
                    velocities.append(0)
                else:
                    notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)
                    velocities.append(element.volume.velocity)

            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                durations.append(element.duration.quarterLength)
                velocities.append(element.volume.velocity)
        counter += 1
        if counter == 10: break

    return notes,durations,velocities

def process_data(data, batch_size=32):
    # get the distinct sets of notes and durations
    notes= data[0]
    durations= data[1]
    velocities= data[2]
    note_types = np.unique(notes)
    duration_types = np.unique(durations)
    velocity_types = np.unique(velocities)
    inp_classes = [note_types, duration_types, velocity_types]

    # make the lookup dictionaries for the 2 features
    dictionaries = [preprocessing.create_dict(note_types), preprocessing.create_dict(duration_types),
                    preprocessing.create_dict(velocity_types)]

    return preprocessing.prepare_batches(notes, durations, velocities,
                                         dictionaries, inp_classes, batch_size)
