# llm_only_baseline/llm_processing.py
import re
# This llm_processing will import from its own config.py inside the llm_only_baseline directory
import config # From llm_only_baseline.config

def get_surah_name_from_number(surah_number: int) -> str:
    """
    Returns the hyphenated Surah name given its number based on the
    SURAH_NO_TO_NAME map from the baseline's config.py.
    Defaults to 'Unknown Surah' if not found.
    """
    return config.SURAH_NO_TO_NAME.get(surah_number, 'Unknown Surah')

def _strip_think_tags(text: str) -> str:
    """Helper to strip any <think>...</think> spans and surrounding whitespace."""
    if text is None:
        return ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def generate_llm_only_formatted_answer(
    user_query: str, # This can be the original or the LLM-preprocessed query
    llm_model,      # The loaded LLM object
    llm_tokenizer,  # The loaded tokenizer object
    max_new_tokens_for_answer: int = 2000 # Max tokens for the generated answer
):
    """
    Generates a formatted answer using only the LLM's internal knowledge.
    The answer structure (intro, Arabic verse, translation, disclaimer) is prompted.
    """
    DISCLAIMER_EN = "For a deeper understanding and specific religious rulings, please consult with a qualified Islamic scholar."
    DISCLAIMER_MS = "Untuk pemahaman yang lebih mendalam dan hukum-hakam agama yang khusus, sila rujuk kepada alim ulama yang berkelayakan."

    if not llm_model or not llm_tokenizer:
        print("LLM (Baseline) for answer generation is not available.")
        return f"LLM for answer generation is not available.\n\n{DISCLAIMER_EN}" # Default disclaimer

    clean_query_for_prompt = _strip_think_tags(user_query)

    # Prompt instructing the LLM to answer based on internal knowledge and follow a specific format
    LLM_ONLY_ANSWER_FORMAT_INSTRUCTIONS = f"""You are an expert Quranic AI assistant. Your SOLE TASK is to generate a direct, factual answer to the user's question based ON YOUR OWN KNOWLEDGE. NO EXTERNAL CONTEXT DOCUMENTS ARE PROVIDED.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1.  **DO NOT** output any of your internal reasoning, decision-making process, or self-correction (e.g., "Okay, let's break this down...", "First, I will...").
2.  **DO NOT** use phrases like "Let me think," "Wait," "Hmm," "Okay, let's tackle this," or similar conversational/reasoning fillers.
3.  Determine the language of the USER'S QUESTION ('{clean_query_for_prompt}'). You MUST answer the question in the SAME LANGUAGE AS THE USER'S QUESTION, except for Quranic verses which should be in ARABIC.
4.  Your entire response MUST be a single, continuous block of text, directly presenting the answer.
5.  Adhere strictly to the content sequence outlined below. Do not add any extra headings, numbering, or commentary unless it's part of the answer itself.
6.  **DO NOT** THINK FOR TRIVIAL THINGS MULTIPLE TIME.
7.  You MUST attempt to provide all 4 pieces of information in the specified order IF you recall any Quranic citations relevant to the query.

User's Question: '{clean_query_for_prompt}'

--- REQUIRED CONTENT SEQUENCE (Answer based on your internal knowledge) ---


First, IN THE SAME LANGUAGE AS THE USER'S QUESTION, provide a comprehensive answer to the user's question ('{clean_query_for_prompt}') based on your internal knowledge of the Quran and Islamic teachings. If you recall relevant Quranic verses that support your answer, you should mention them in this part and then cite them fully in Part 2.

Second, IF AND ONLY IF you recalled specific Quranic verses relevant to the answer in Part 1, cite them here IN ARABIC.
- For EACH cited verse or range of verses, use this format based on the query language:
  - If query in English: "Based on Quran, surah [SurahName (e.g., Al-Baqarah)] ([SurahNo]:[AyahNo(s) e.g., 2:255 or 2:1-5]),
    \"[Full Arabic Verse Text you recall. If multiple verses, clearly number them if appropriate, e.g., (1) ... (2) ... (3) ...]\""
  - If query in Malay: "Berdasarkan Al-Quran, surah [SurahName] ([SurahNo]:[AyahNo(s)]),
    \"[Full Arabic Verse Text you recall. Jika beberapa ayat, nomborkannya jika sesuai, cth., (١) ... (٢) ... (٣) ...]\"")
- If you do not recall any specific verses to cite for Part 1, OMIT THIS ENTIRE PART 2 AND PART 3.

Third, IF AND ONLY IF a verse was cited in Part 2, immediately follow EACH Arabic verse (or block of verses) with its translation IN THE SAME LANGUAGE AS THE USER'S QUESTION. Provide the translation based on your internal knowledge.
- Format based on the query language:
  - If query in English: "which means: \"[Full Translation Text you recall]\""
  - If query in Malay: "yang bermaksud: \"[Full Translation Text you recall]\"")
- If a translation is unavailable from your knowledge for a cited verse, state: "Translation in the language of your question for [SurahName] ([SurahNo]:[AyahNo(s)]) is not available from my current knowledge."
- If no verse was cited in Part 2, OMIT THIS ENTIRE PART 3.


Fourth, Conclude your ENTIRE response with the EXACT disclaimer sentence appropriate for the language of the user's question ('{clean_query_for_prompt}'). Use ONLY ONE of the following, and it must be the absolute end of your response:
- If user's question appears to be in English: "{DISCLAIMER_EN}"
- If user's question appears to be in Malay: "{DISCLAIMER_MS}"
(If unsure of the language, default to the English disclaimer.)

--- END OF INSTRUCTIONS ---
Begin your response directly with the first piece of information.
"""
    final_llm_prompt_content = LLM_ONLY_ANSWER_FORMAT_INSTRUCTIONS
    # System message to reinforce strict adherence to formatting and no reasoning.
    system_message_content = (
        "You are an expert Quranic AI assistant. Your absolute priority is to generate a direct, factual, and structured answer based on your internal knowledge. "
        "DO NOT output any internal reasoning, self-correction, or conversational text. Follow the user's formatting instructions PRECISELY."
    )
    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": final_llm_prompt_content}
    ]

    try:
        prompt_for_model = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = llm_tokenizer(
            [prompt_for_model], return_tensors="pt"
        ).to(llm_model.device)

        temperature = 0.05 # Keep low for factual recall, can be slightly higher if more generative style needed
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_answer,
            temperature=temperature,
            top_p=0.3, # Focused sampling
            do_sample=True if temperature > 0 else False, # Sample if temp > 0
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        answer_text_raw = llm_tokenizer.decode(
            generated_ids[0, model_inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        
        # Strip <think> tags first
        answer_text = _strip_think_tags(answer_text_raw)
        
        # Strip common reasoning prefixes
        reasoning_prefixes_to_strip = [
            "Okay, let's", "Alright, here's", "Let me try to", "Thinking...",
            "First, I will", "Here is the breakdown:", "Okay, I will answer",
            "Okay, here is the answer in the requested format:",
            "Here's the answer based on my internal knowledge:",
            "Based on my internal knowledge:" # Common preamble
        ]
        for prefix in reasoning_prefixes_to_strip:
            if answer_text.lower().startswith(prefix.lower()):
                answer_text = answer_text[len(prefix):].lstrip(" :.\n")
                break # Remove only the first matched prefix
        
        # Strip "Final Answer:" type prefixes
        if answer_text.lower().startswith("final answer:"):
            answer_text = re.split(r"final answer:", answer_text, maxsplit=1, flags=re.IGNORECASE)[-1].strip()

        # Disclaimer Enforcement - Ensure it's the absolute last thing
        normalized_answer_for_disclaimer_check = ' '.join(answer_text.lower().split())
        normalized_disclaimer_en = ' '.join(DISCLAIMER_EN.lower().split())
        normalized_disclaimer_ms = ' '.join(DISCLAIMER_MS.lower().split())

        is_en_disclaimer_at_end = normalized_answer_for_disclaimer_check.endswith(normalized_disclaimer_en)
        is_ms_disclaimer_at_end = normalized_answer_for_disclaimer_check.endswith(normalized_disclaimer_ms)

        if not (is_en_disclaimer_at_end or is_ms_disclaimer_at_end):
            print("Warning (LLM-only Baseline): LLM did not include the disclaimer correctly or not at the very end. Appending.")
            
            disclaimer_to_append = DISCLAIMER_EN # Default to English
            # Simple heuristic for query language to choose disclaimer
            if any(k_word in user_query.lower() for k_word in ["apa", "makna", "sila", "bagaimana", "kenapa", "terangkan", "maksud", "berikan", "jelaskan", "bolehkah", "bagaimanakah"]):
                disclaimer_to_append = DISCLAIMER_MS
            
            # Attempt to remove any partial/incorrect disclaimers before appending the correct one
            answer_text_lower_for_cleaning = answer_text.lower()
            if normalized_disclaimer_en[:30] in answer_text_lower_for_cleaning and disclaimer_to_append == DISCLAIMER_MS: # If English disclaimer found but Malay needed
                 idx_en = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_en[:20]) # Find start of partial Eng
                 if idx_en != -1: answer_text = answer_text[:idx_en].strip()
            elif normalized_disclaimer_ms[:30] in answer_text_lower_for_cleaning and disclaimer_to_append == DISCLAIMER_EN: # If Malay disclaimer found but Eng needed
                 idx_ms = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_ms[:20]) # Find start of partial Malay
                 if idx_ms != -1: answer_text = answer_text[:idx_ms].strip()
            # Also remove if the *correct* disclaimer is present but not at the very end
            elif disclaimer_to_append == DISCLAIMER_EN and normalized_disclaimer_en in normalized_answer_for_disclaimer_check and not is_en_disclaimer_at_end:
                 idx_en = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_en[:20])
                 if idx_en != -1: answer_text = answer_text[:idx_en].strip()
            elif disclaimer_to_append == DISCLAIMER_MS and normalized_disclaimer_ms in normalized_answer_for_disclaimer_check and not is_ms_disclaimer_at_end:
                 idx_ms = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_ms[:20])
                 if idx_ms != -1: answer_text = answer_text[:idx_ms].strip()


            # Ensure proper spacing before appending
            if answer_text and not answer_text.endswith(("\n\n", "\n")):
                answer_text += "\n\n" # Add two newlines if none
            elif answer_text and not answer_text.endswith("\n\n"): # if ends with one newline
                answer_text += "\n" # Add one more newline
            answer_text += disclaimer_to_append
        
        print(f"LLM-Only Formatted Answer (Baseline) - Raw LLM: \"{answer_text_raw[:150]}...\" -> Final: \"{answer_text[:150]}...\"")
        return answer_text # Returns only the answer string

    except Exception as e:
        print(f"Error during LLM-only structured answer generation (Baseline): {e}")
        # Determine appropriate disclaimer based on query language heuristic
        disclaimer_to_append_on_error = DISCLAIMER_EN
        if any(k_word in user_query.lower() for k_word in ["apa", "makna", "sila", "bagaimana", "kenapa", "terangkan", "maksud", "berikan", "jelaskan", "bolehkah", "bagaimanakah"]):
            disclaimer_to_append_on_error = DISCLAIMER_MS
        return f"An error occurred while generating the LLM-only structured answer.\n\n{disclaimer_to_append_on_error}"

