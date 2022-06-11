for stfile in $(ls snippets/*.py)
do
    streamlit run $stfile &  # Push call to a subshell running in the background, and thus not blocking
    echo "-----------------------------------------------------------------------------"
    echo "Press enter to kill the current streamlit process and launch the next snippet"
    echo "-----------------------------------------------------------------------------"
    read _  # Wait for user input
    pkill -P $$  # kill all shell decendants (`streamlit run` is the only one)
done
