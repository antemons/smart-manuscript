# SMART MANUSCRIPT

This software transcripts handwritten manuscripts or digitizer pen input.

It recognizes on-line handwriting (generated e.g. by smart pens) and can not read scanned pages.

## Getting Started

## SETUP

 1. Install [Tensorflow](https://www.tensorflow.org/get_started/os_setup) >= 0.12.0

 2. Install numpy, scipy, pylab, svgpathtools, svgwrite

 3. Optional: train the graph (or use the one in sample_graph)

## USAGE

### Transcript handwritten notes

Transcript the handwritten (generated e.g. by smart pens)

    python3 smart_manuscript/transcript.py --file=sample_text/The_Zen_of_Python.svg

If you can export only pdf-files, convert this by

  pdftocairo -svg filename.pdf

### Handwritten input

A simple application that transcript handwritten input (e.g. from a digitizer pen):

    python3 smart_manuscript/application.py

The input will be copied into the clipboard.

### Train new model

A new model can be trained by

    python3 smart_manuscript/train.py --name=new_model --build_data=0

You can use the [IAMonDo-db-1.0](http://www.iapr-tc11.org/dataset/IAMonDo/IAMonDo-db-1.0.tar.gz) database
to train and validate the model. Place the unzipped folder IAMonDo-db-1.0 in the folder data. You may use also your personal handwritten notes, analogously to the one in the directory sample_text (see --train_my_writing)

## AUTHOR

Daniel Vorberg

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## CONTACT

Please send bug reports, patches, and other feedback to: dv(at)pks.mpg.de
