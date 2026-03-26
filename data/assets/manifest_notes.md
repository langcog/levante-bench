# Manifest development notes

- full_prompt contains the prompt for the trial. It has some possible placeholders:
    - <prompt_phrase> should be substituted with the value from prompt_phrase
    - <prompt_image> should be substituted with the corresponding prompt_image file (requires some filepath modification)
    - <optionX> should be substituted with a randomised ordering of {answer, response_alternatives}
    - <imageX> should be substituted with a randomised ordering of the corresponding image files from {answer, response_alternatives} (also requires some filepath modification)
- full_prompt was made mostly by translating the original prompt into more model-appropriate language. Each prompt contains the trial instructions, a specification of the answer format, and the response option set.
- Currently, full_prompt is set up under the multi-image setting. Some math items only require a single image or no image, and are straightforwardly translatable to the single-/no image settings, but more work is needed to translate the other items.

- Mental rotation items have been uniquified by concatenating the item_uid with items (underscore separated).
- Math number line identification items require number line images that have yet to be generated.
- Math number line response items were not implemented. 
- SDS was not implemented.
- TOM was implemented without using the scene images.

# Task construction decision - TODO
There are a few ways of constructing tasks:
- Use manifest.csv, which has the advantage of standardisation and cleaned up language + centralisation.
- Use the corpus files, which has the advantage of being directly linked to the most recent asset bank.
Note that the latter has an issue whereby updating the corpus bank means that previous versions are not accessible (no archiving of assets). 
We could also consider writing a translator that converts from the corpus format into a manifest format, which we could archive.