"""Human response distribution comparison (hard-prediction path).

This module computes per-trial metrics that compare a model's predicted label
against human response proportions produced by ``download_levante_data.R``.

No model probability outputs are required: the model's prediction is treated
as a single hard label, then compared with the human distribution.

Notes on human data structure
------------------------------
The R script builds the option key as [answer_col, *parsed_distractors].
For tasks like egma-math the ``answer`` column in Redivis IS the correct
answer, so the first canonical option (image1) carries the highest proportion
on easy items.  For tasks like vocab the ``answer`` column stores a distractor
word (the Redivis vocab item format differs from math), so the correct target
word rarely appears in the option key and its mapped proportion will be 0.0.
Callers should interpret ``human_correct_prop = 0.0`` for vocab as "data not
available", not as "no children answered correctly".
"""

from typing import Optional


def annotate_human_metrics(result: dict, human_entry: Optional[dict]) -> dict:
    """Compute human-comparison metrics and merge them into *result*.

    Parameters
    ----------
    result:
        Output dict from ``VLMModel.evaluate_trial`` **extended** to also
        contain ``options`` (list[str], shuffled benchmark word order) and
        ``option_labels`` (list[str], e.g. ["A","B","C","D"]).
    human_entry:
        Single item from ``load_human_proportions`` output, or None when
        human data is unavailable for this item_uid.

    Returns
    -------
    The same *result* dict with four new keys added:

    human_correct_prop : float | None
        Proportion of humans who chose the correct answer according to the
        option key (see note above — may be 0.0 / unreliable for vocab).
    human_predicted_prop : float | None
        Proportion of humans who chose the same option as the model.
    human_plurality_label : str | None
        The benchmark label (A/B/C/D) that received the highest human
        proportion; the "most common human answer".
    human_plurality_agrees_model : bool | None
        True when the model's prediction equals human_plurality_label.
    """
    _null = {
        "human_correct_prop": None,
        "human_predicted_prop": None,
        "human_plurality_label": None,
        "human_plurality_agrees_model": None,
    }

    if human_entry is None:
        result.update(_null)
        return result

    options: list[str] = result.get("options") or []
    option_labels: list[str] = result.get("option_labels") or []
    predicted_label: Optional[str] = result.get("predicted_label")
    correct_label: Optional[str] = result.get("correct_label")

    if not options or not option_labels:
        result.update(_null)
        return result

    canonical: list[str] = human_entry.get("canonical_options", [])
    proportions: list[float] = human_entry.get("proportions", [])

    # word (lowercase, stripped) → human proportion
    word_to_prop: dict[str, float] = {}
    for word, prop in zip(canonical, proportions):
        key = str(word).strip().lower()
        if key:
            word_to_prop[key] = float(prop)

    # Map benchmark labels → human proportions via word matching
    label_props: dict[str, float] = {}
    for label, word in zip(option_labels, options):
        label_props[label] = word_to_prop.get(str(word).strip().lower(), 0.0)

    human_correct_prop = label_props.get(correct_label) if correct_label else None
    human_predicted_prop = (
        label_props.get(predicted_label) if predicted_label else None
    )

    # Plurality: label with highest human proportion
    human_plurality_label: Optional[str] = (
        max(label_props, key=label_props.__getitem__) if label_props else None
    )

    human_plurality_agrees_model: Optional[bool] = (
        predicted_label == human_plurality_label
        if (predicted_label is not None and human_plurality_label is not None)
        else None
    )

    result.update(
        {
            "human_correct_prop": human_correct_prop,
            "human_predicted_prop": human_predicted_prop,
            "human_plurality_label": human_plurality_label,
            "human_plurality_agrees_model": human_plurality_agrees_model,
        }
    )
    return result
