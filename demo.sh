#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#make
#if [ ! -e text8 ]; then
#  if hash wget 2>/dev/null; then
#    wget http://mattmahoney.net/dc/text8.zip
#  else
#    curl -O http://mattmahoney.net/dc/text8.zip
#  fi
#  unzip text8.zip
#  rm text8.zip
#fi

CORPUS='/home/eirini/glove/CorpusNew/Krapivin2009_all'
#$1
#/home/eirini/glove/J-47.txt
VOCAB_FILE="vocab-Krapivin2009-all-200.txt"
#$2$3$4"
#vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
#cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
#cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE="vectors-Krapivin2009-all-200"
#$2$3$4"
#vectors
VERBOSE=2
MEMORY=4
VOCAB_MIN_COUNT=1
VECTOR_SIZE=200
#$3
#1000
MAX_ITER=50
#$4
#5000
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=100

echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ python eval/python/evaluate.py"
       python eval/python/evaluate.py
   fi
fi
