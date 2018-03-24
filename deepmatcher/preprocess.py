
WORD_EMBEDDING_PATHS = {

}

def preprocess(table, word_embedding_type='fasttext', word_embedding_path=None,
        disable_tokenizer=False, disable_lowercasing=False, save_path=None):
    """
    This function debugs the blocker output and reports a list of potential
    matches that are discarded by a blocker (or a blocker sequence).
    Specifically,  this function takes in the two input tables for
    matching and the candidate set returned by a blocker (or a blocker
    sequence), and produces a list of tuple pairs which are rejected by the
    blocker but with high potential of being true matches.

    Args:
        table (DataFrame): Dataframe containing pairs of entities. Table must
                           be validated if specified. May optionally contain
                           a "label" column. If None, `dataset_path` must be
                           specified and must contain a valid preprocessed
                           dataset. (Defaults to None)
        word_embedding_type (string): One of "Glove", "Word2vec" or
                                      "FastText". (Defaults to "FastText")
        word_embedding_path (string): Path to word embedding file.
                                      (Defaults to None)
        disable_tokenizer (boolean): Defaults to False
        disable_lowercasing (boolean): Defaults to False
        dataset_path (string): Path to store preprocessed data. If None,
                               `table` must be a validated table.
                               (Defaults to None)
        batch_size (int): Batch size. (Defaults to 32)

    Returns:
        An object of class deepmatcher.Dataset.
        Note: deepmatcher.Dataset is a thin wrapper around tf.Dataset that
              also stores the raw preprocessed table. This can be obtained by
              calling dataset.get_raw_table()
    Raises:
    Examples:
    """

    
