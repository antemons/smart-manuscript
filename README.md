# SMART MANUSCRIPT

This software recognizes (digitizes) handwritten manuscripts and digitizer-pen input.

Note, it can only transcribe online handwriting (generated e.g. by smart pens) but no scanned pages.

## Getting Started

### Setup

 1. (Create Virtual Environment)

        virtualenv env -p /usr/bin/python3
        source env/bin/activate

 2. Install requirements:

        pip install -r requirements.txt

 3. Install this software

        python setup.py install

 4. Optional: train a new model (or use the default one)

### Usage

#### Recognize handwritten notes

Recognize a handwritten file (of type either PDF or SVG, generated e.g. by a smart pen) and
generate a PDF which is searchable and where text can be copied from.

    transcribe data/sample_text/The_Zen_of_Python.pdf output.pdf

#### Handwritten input

A simple application that transcribes handwritten input (e.g. from a digitizer pen):

    manuscript-writer

The input will be copied into the clipboard.

#### Train new model

 1. Download and extract data:

    a. You can use the [IAMonDo-db-1.0](http://www.iapr-tc11.org/dataset/IAMonDo/IAMonDo-db-1.0.tar.gz) database to train and validate the model. Place the unzipped folder "IAMonDo-db-1.0" in the folder data.

    b.  You can use the [IBM_UB_1](https://cubs.buffalo.edu/research/50:hwdata) database to train and validate the model. Place the folder "IBM_UB_1" in the folder data.

    c. You may use also your personal handwritten notes, analogously to the one in the directory smart_manuscript/data/sample_text. Place the files in the folder "data/my_handwriting".

 2. Create the preprocessed records:

        python -m smartmanuscript.records \
           --path="records" \
           --iam_on_do_path="./data/IAMonDo-db-1.0/" \
           --ibm_ub_path="./data/IBM_UB_1/query" \
           --my_handwriting_train_path="./data/my_handwriting/train/" \
           --my_handwriting_test_path="./data/my_handwriting/test/"

    The new folder "records" should contain a "test" and a "train" folder

 3. Train new model:

        python train_model.py train --name=my_model

## Example

<a href="smartmanuscript/data/sample_text/The_Zen_of_Python.pdf">
  <img src="smartmanuscript/data/sample_text/The_Zen_of_Python.png" width="75%" height="75%">
</a>

Transcription:

<pre>
The zen of Python.
by tim Peters
Beantiful is better than ugly.
Explicit is better than implicit.
Simple is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the mles.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of anbiguity, refuse the temptation to guess.
There should be one - and preferable onty dne
- obvious way to do it.
Although that way may not be obvious at
first unless youire Dutch.
Now is better than never.
Although never is often better than right now.
if the implementation is hard to explain, it's a bad idea.
It the implementation is easy to explain, it may be
a good idea.
Namespaces are one hanking great idea -
let's do more of thosel
</pre>

## Author

Daniel Vorberg

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## Contact

Please send bug reports, patches, and other feedback to: dv(at)pks.mpg.de
