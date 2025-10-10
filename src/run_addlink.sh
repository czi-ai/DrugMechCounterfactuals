#!/bin/bash
#
#  Run all the Add-Link queries against the 4 ChatGPT models
#
#  To run:
#	$ source $PRODIR/.venv/bin/activate
#	(dmcf) $ cd $PRODIR/src
#	(dmcf) $ ./run_addlink.sh
#

CMD=`basename $0`

start_date=`date`
echo "Start: ${start_date}"


# -- Paths

DATA_DIR=../Data/Counterfactuals

DEST_DIR=../Data/Sessions/Models


# -- Data files

POS_DATA_FILES=($DATA_DIR/AddLink_pos_*.json)

NEG_DATA_FILES=($DATA_DIR/AddLink_neg_*.json)

DATA_FILES=("${POS_DATA_FILES[@]}"  "${NEG_DATA_FILES[@]}")

echo
echo "Add-Link Data files:"
echo "   nbr +ive Link files = ${#POS_DATA_FILES[@]}"
echo "   nbr -ive Link files = ${#NEG_DATA_FILES[@]}"
echo "   Total nbr files = ${#DATA_FILES[@]}"
echo


# -- Options

MODELS=("4o" "o3" "o3-mini" "o4-mini")

# ... Open-world and Closed-world
OPTIONS=("" "-k")


# -- Nested loop

for model in "${MODELS[@]}"; do

    DEST="$DEST_DIR/$model"

    if [ ! -d "${DEST}" ]; then
        echo "Creating dir:"
        mkdir -pv $DEST
    fi

    for dataf in "${DATA_FILES[@]}"; do

        dataf_base=`basename ${dataf}`
        rootf="${dataf_base%.*}"

        for opt in "${OPTIONS[@]}"; do

            if [ -z $opt ]; then
                outroot="$rootf"
            else
                outroot="${rootf}${opt}"
            fi

            jsonf="${DEST}/${outroot}.json"
            logf="${DEST}/${outroot}_log.txt"

            cmd_opts="-m ${model} ${opt}"

            echo "----------------------------------------"
            echo "opts = ${cmd_opts}"
            echo "JSON file = ${jsonf}"
            echo "LOG file = ${logf}"
            echo

            python -m drugmechcf.llmx.test_addlink batch ${cmd_opts} ${dataf} ${jsonf} > ${logf} 2>&1

            echo "${CMD}:  ${cmd_opts} ${dataf} ... completed"
            echo

        done

    done

done

echo "-- ${CMD} --"
echo "Started at: ${start_date}"
echo "All Completed at:" `date`
