import json
import pandas


def predictions_report_kgxboard(
        top_predictions_dict,
        triples,
        config,
        dataset,
        split,
        output_filename,
        use_strings=False,
):
    """
    Dumps the JSON file with top k predictions as required by KGxBoard

    :param top_predictions: dict of dicts containing top scores, entities,
    rankings and ties
    :param triples: tensor of size nx3 with the n triples used for evaluation
    :param config: job's config object
    :param dataset: job's dataset object
    :param output_filename: name of file to dump content
    :param use_strings: print triples and predictions with readable strings, if
    available
    """

    config.log("Dumping top predictions in KGxBoard format")
    # set names to be dataset IDs or readable strings
    if use_strings:
        ent_names = dataset.entity_strings
        rel_names = dataset.relation_strings
    else:
        ent_names = dataset.entity_ids
        rel_names = dataset.relation_ids

    entities = top_predictions_dict["entities"]
    scores = top_predictions_dict["scores"]
    rankings = top_predictions_dict["rankings"]
    # TODO Printing number of ties for each prediction set is a good idea
    # ties = top_predictions_dict["ties"]

    # construct examples
    examples = []
    for triple_pos in range(len(triples)):
        # create entry for each example and prediction task
        for slot, task in zip(["sub", "obj"], ["head", "tail"]):
            example_dict = {}
            example_dict["gold_head"] = ent_names(triples[triple_pos, 0]).item()
            example_dict["gold_predicate"] = rel_names(
                triples[triple_pos, 1]
            ).item()
            example_dict["gold_tail"] = ent_names(triples[triple_pos, 2]).item()
            example_dict["predict"] = task
            example_dict["predictions"] = \
                ent_names(entities[slot][triple_pos, :].int()).tolist()
            example_dict["scores"] = scores[slot][triple_pos, :].tolist()
            example_dict["true_rank"] = rankings[slot][triple_pos].item()
            # example_dict["ties"] = ties[slot][triple_pos].item()
            example_dict["example_id"] = triple_pos

            # add current example
            examples.append(example_dict)

    # TODO stuff that is a good idea to put in metadata
    #   1. Filter splits information
    #   2. Tie handling policy
    #   3. Framework, i.e. libkge
    # for now, no metadata
    # # construct main dict
    # main_dict = {
    #     "metadata": {
    #         "model": config.get("model"),
    #         "dataset": config.get("dataset.name"),
    #         "split": split,
    #     },
    #     "examples": examples
    # }
    main_dict = examples

    # dump main dict to json
    with open(output_filename, "w") as file:
        file.write(json.dumps(main_dict, indent=2))
    k = scores["sub"].size()[1]
    config.log(
        "Dumped top {} predictions to file: {}".format(k, output_filename)
    )

    # dump config settings
    flattened_config_dict = pandas.json_normalize(
        config.options, sep="."
    ).to_dict(orient="records")[0]

    # remove unwanted config settings
    train_type = config.get("train.type")
    model = config.get("model")
    if model == "reciprocal_relations_model":
        model = config.get("reciprocal_relations_model.base_model.type")
    wanted_keys = {
        "random_seed",
        "dataset.name",
        "model",
        "lookup_embedder",
        model,
        "train",
        train_type,
        "eval",
        "entity_ranking",
        "valid",
    }
    config_dict = {}
    for k, v in flattened_config_dict.items():
        root_key = k.split(".")[0]
        if root_key in wanted_keys:
            config_dict[k] = v

    # assumes given output filename has an extension
    output_filename_split = output_filename.split(".")
    output_filename = "".join(output_filename_split[:-1]) + \
                      "_hyperparameters." + output_filename_split[-1]
    with open(output_filename, "w") as file:
        file.write(json.dumps(config_dict, indent=2))
    config.log(
        "Dumped flattened config settings to file: {}".format(output_filename)
    )
