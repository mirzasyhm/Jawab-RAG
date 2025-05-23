# llm_processing.py
import re
from config.config import SURAH_NO_TO_NAME # Import the new reverse mapping

def get_surah_name_from_number(surah_number: int) -> str:
    """
    Returns the hyphenated Surah name given its number.
    Defaults to 'Unknown Surah' if not found.
    """
    return SURAH_NO_TO_NAME.get(surah_number, 'Unknown Surah')

def _strip_think_tags(text: str) -> str:
    """Helper to strip any <think>...</think> spans."""
    if text is None:
        return ""
    # This will also remove any leading/trailing whitespace around the removed think block
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def preprocess_user_query_with_llm(original_query: str,
                                    llm_model, # e.g., Qwen model
                                    llm_tokenizer,
                                    max_new_tokens_for_preprocessing: int = 100): # From user cell, was 1000, adjusted to 100
    """
    Uses an LLM to refine the original user query.
    The goal is to make it clearer for subsequent routing and retrieval,
    and to explicitly format any Surah:Ayah references if understood.
    Strips off any thinking/reasoning preamble from the LLM's raw output.
    """
    if not llm_model or not llm_tokenizer:
        print("Preprocessing LLM not available. Using original query.")
        return original_query

    system_prompt_template = """You are an AI assistant that refines user questions about the Quran to make them clearer and more effective for a retrieval system.
Your output should be a single, optimized query.

Key Instructions:
1.  If the user mentions a specific Surah and Ayah (e.g., "verse 22 of surah al baqarah", "al-baqarah 22", "2:22 al baqarah"), your output query MUST explicitly include this reference in a standardized format, ideally like "verse AYAH_NUMBER surah SURAH_NAME_HYPHENATED". Example: "verse 22 surah al-baqarah". Use the hyphenated Surah name.
2.  If the user asks a general question, answer the question based on your external knowledge then rephrase it clearly and concisely for a search system.
3.  ONLY MENTION the specific surah and verse if you are CONFIDENT of it.
4.  If you are not sure of the answer to the question based on your external knowledge. JUST refined the query without the verse reference.
5.  If the question is in a language other than English (e.g., Malay, Arabic), try to preserve the original language in your output query unless rephrasing for clarity inherently changes it slightly.
6.  The output should ONLY be the refined query. No explanations, no preamble.
7.  DO NOT OVERTHINK

Examples:
Original: "tell me about al fatihah ayah 2"
Refined Query: verse 2 surah al-fatihah

Original: "What does Quran say about giving to the poor?"
Refined Query: Quranic teachings on charity and helping the poor

Original: "makna ayat kursi"
Refined Query: makna Ayatul Kursi quran 2:255

Original: "surah baqarah verse 50 please"
Refined Query: verse 50 surah al-baqarah

Original: "What is the main message of Surah Al-Ikhlas regarding Allah's oneness?"
Refined Query: Surah al-ikhlas (112:1-4) Allah's oneness

Original: "According to Al-Fatihah, who is the Lord of the worlds?"
Refined Query: Al-Fatihah surah_no 1. Lord of the worlds

Original: "Sebutkan ciri-ciri orang yang berjaya menurut Surah Al-Mu'minun ayat 1 hingga 5 dalam bahasa Melayu."
Refined Query: Al-Mu'minun (23:1-5) ciri-ciri orang berjaya

Now, process the following:
Original User Question: "{user_query}"
Refined Query:
"""
    final_system_prompt = system_prompt_template.format(user_query=original_query)
    messages = [{"role": "system", "content": final_system_prompt}] # Per original code, using system role

    try:
        prompt_for_model = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([prompt_for_model], return_tensors="pt").to(llm_model.device)

        temperature = 0.05
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_preprocessing,
            temperature=temperature,
            top_p=0.7,
            do_sample=True if temperature > 0 else False, # Corrected logic based on temperature
            pad_token_id=llm_tokenizer.eos_token_id
        )
        # The '...' in user's paste indicated a continuation or truncation.
        # Assuming the decode line is complete and the if condition is next.
        raw_llm_output_text = llm_tokenizer.decode(generated_ids[0, model_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if not raw_llm_output_text:
            print("Preprocessing LLM returned empty. Using original query.")
            return original_query

        # --- STRIPPING LOGIC from user's paste ---
        cleaned_after_think_tags = _strip_think_tags(raw_llm_output_text)
        refined_query_final = cleaned_after_think_tags

        possible_prefixes = [
            r"Refined Query:\s*",
            r"Optimized Query:\s*",
        ]
        query_extracted = False
        for prefix_regex in possible_prefixes:
            match = re.search(prefix_regex + r'(.*)', cleaned_after_think_tags, flags=re.IGNORECASE | re.DOTALL)
            if match:
                refined_query_final = match.group(1).strip()
                query_extracted = True
                break
        
        if not query_extracted:
            # Fallback: if no prefix found, consider the last non-empty line if multiple lines exist.
            # This is heuristic, LLM consistency is preferred.
            lines = cleaned_after_think_tags.split('\n')
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line: # If last line is not empty
                    # A simple heuristic: if the full text is much longer than the last line,
                    # and the last line is reasonably short, it might be the intended query.
                    if len(cleaned_after_think_tags) > len(last_line) + 20 and len(last_line) < 150 :
                         refined_query_final = last_line
                    # else keep refined_query_final as cleaned_after_think_tags
            # if only one line, refined_query_final remains cleaned_after_think_tags

        refined_query_final = refined_query_final.strip('"\'\n ')

        if not refined_query_final:
             print(f"Preprocessing LLM output was stripped to empty. Raw output: '{raw_llm_output_text}'. Using original query.")
             return original_query

        print(f"Original Query: \"{original_query}\" -> LLM Raw Output (first 100 chars): \"{raw_llm_output_text[:100]}...\" -> Final Preprocessed Query: \"{refined_query_final}\"")
        return refined_query_final
    except Exception as e:
        print(f"Error during query preprocessing with LLM: {e}. Using original query.")
        return original_query


def generate_formatted_answer_with_llm(
    original_user_query: str,
    retrieved_documents: list,  # List of Haystack Document objects
    llm_model,
    llm_tokenizer,
    max_new_tokens_for_answer: int = 2000
):
    DISCLAIMER_EN = "For a deeper understanding and specific religious rulings, please consult with a qualified Islamic scholar."
    DISCLAIMER_MS = "Untuk pemahaman yang lebih mendalam dan hukum-hakam agama yang khusus, sila rujuk kepada alim ulama yang berkelayakan."

    if not llm_model or not llm_tokenizer:
        print("Formatted answer generation LLM not available.")
        return f"LLM for answer generation is not available.\n\n{DISCLAIMER_EN}", []

    clean_query_for_prompt = _strip_think_tags(original_user_query) # Use the already preprocessed query if available

    DETAILED_ANSWER_FORMAT_INSTRUCTIONS = f"""You are an expert Quranic AI assistant. Your SOLE TASK is to generate a direct, factual answer to the user's question based on your knowledge and the provided context.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1.  **DO NOT** output any of your internal reasoning, decision-making process, or self-correction.
2.  **DO NOT** use phrases like "Let me think," "Wait," "Hmm," "Okay, let's tackle this," or similar conversational/reasoning fillers.
3.  Find out the USER'S QUESTION'S LANGUAGE. You MUST answer the question in the SAME LANGUAGE AS THE USER'S QUESTION except for the ARABIC QURAN CITATION.
4.  Your entire response MUST be a single, continuous block of text, directly presenting the answer.
5.  Adhere strictly to the content sequence outlined below. Do not add any extra headings, numbering, or commentary.
6.  **DO NOT** THINK FOR TRIVIAL THINGS MULTIPLE TIME.
7. You MUST provide all 4 pieces of information in the order IF THERE ARE QURAN CITATION.

User's Original Question: '{clean_query_for_prompt}'

Provided Context Documents (verses, translations, and any associated metadata):
{{context_documents}}

--- REQUIRED CONTENT SEQUENCE ---

First, IN THE SAME LANGUAGE AS THE USER'S QUESTION provide a comprehensive answer to the user's question ('{clean_query_for_prompt}') based on the provided context documents. If the 'Provided Context Documents' contain relevant information or verses, you MUST integrate and reference them to support or illustrate your answer.

Second, cite ALL relevant Quranic verses (IN ARABIC) from the 'Provided Context Documents' that are foundational to your answer or directly address the query. For EACH cited verse, use this format based on the query language:
If the query in english use this format: "Based on Quran, surah [SurahName] ([SurahNo]:[AyahNo]-[AyahNo (if more than 1 ayah)]),
\"[Full Arabic Verse Text From Context Documents (If more than 1 verse, put the verse number at the end of each verse)]\"" or
if the query was in Malay use this format: "Berdasarkan Al-Quran, surah [SurahName] ([SurahNo]:[AyahNo]-[AyahNo (if more than 1 ayah)]), \"[Full Arabic Verse Text From Context Documents (If more than 1 verse, put the verse number at the end of each verse)]\"")
(Include Ayah numbers within the Arabic text if present in source, e.g., "...(1)...(2)...").
If no verses from context are cited, OMIT this part and the 'Third' part (translation).

Third (IF a verse was cited in the 'Second' part), immediately follow EACH Arabic verse (or verses if the verses are multiple in sequence) with its translation in the SAME LANGUAGE AS THE USER'S QUESTION (infer from '{clean_query_for_prompt}'). Quote verbatim from 'Provided Context Documents' depends on the query language either english or malay. Format:
If query in english use this: "which means: \"[Full Translation Text From Context Documents]\"" or
If query in Malay use this: "yang bermaksud: \"[Full Translation Text From Context Documents]\"")
If a translation is unavailable, state: "Translation in the language of your question for [SurahName] ([SurahNo]:[AyahNo]) is not available in the provided context." If no verse was cited, OMIT this part.

Fourth, conclude your ENTIRE response with the EXACT disclaimer sentence appropriate for the language of the user's question ('{clean_query_for_prompt}'). Use ONLY ONE of the following:
- If user's question appears English: "{DISCLAIMER_EN}"
- If user's question appears Malay: "{DISCLAIMER_MS}"
This disclaimer must be the absolute end of your response.

--- END OF INSTRUCTIONS ---
Begin your response directly with the first piece of information.
"""
    if not retrieved_documents:
        context_str = (
            "No specific Quranic context documents were retrieved for this query. "
            "For the first part of your response, answer based on your general knowledge. For the second and third parts (verse citation and translation), indicate they cannot be fulfilled from the provided context."
        )
    else:
        context_str = ""
        for i, doc in enumerate(retrieved_documents):
            arabic_text = _strip_think_tags(doc.content) # Assuming doc.content is Arabic
            surah_no_meta = doc.meta.get('surah_no', 'N/A')
            ayah_no_meta = doc.meta.get('ayah_no', 'N/A')
            
            surah_name_en_meta = 'Unknown Surah'
            if isinstance(surah_no_meta, (int, str)) and str(surah_no_meta).isdigit():
                surah_name_en_meta = get_surah_name_from_number(int(surah_no_meta))
            elif doc.meta.get('surah_name_en'): # Fallback to surah_name_en if present
                 surah_name_en_meta = doc.meta.get('surah_name_en')


            trans_en = doc.meta.get('translation_en', 'English translation not provided in this context document.')
            trans_ms = doc.meta.get('translation_ms', 'Malay translation not provided in this context document.')

            context_str += (
                f"Context Document {i+1} (ID: {doc.id}):\n"
                f"  Surah Name (English): {surah_name_en_meta}\n"
                f"  Surah Number: {surah_no_meta}\n"
                f"  Ayah Number(s): {ayah_no_meta}\n"
                f"  Arabic Text: \"{arabic_text}\"\n"
                f"  English Translation: \"{trans_en}\"\n"
                f"  Malay Translation: \"{trans_ms}\"\n\n"
            )
    context_str = _strip_think_tags(context_str.strip())

    final_llm_prompt_content = DETAILED_ANSWER_FORMAT_INSTRUCTIONS.format(
        context_documents=context_str
    )
    system_message_content = (
        "You are an expert Quranic AI assistant. Your absolute priority is to generate a direct, factual, and structured answer. "
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

        temperature = 0.05
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_answer,
            temperature=temperature,
            top_p=0.3,
            do_sample=True if temperature > 0 else False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        answer_text_raw = llm_tokenizer.decode(
            generated_ids[0, model_inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        
        answer_text = _strip_think_tags(answer_text_raw)
        
        # Additional stripping logic from user's paste
        reasoning_prefixes_to_strip = ["Okay, let's", "Alright, here's", "Let me try to", "Thinking...", "First, I will", "Here is the breakdown:"]
        for prefix in reasoning_prefixes_to_strip:
            if answer_text.lower().startswith(prefix.lower()):
                answer_text = answer_text[len(prefix):].lstrip(" :.\n")
                break
        
        if answer_text.lower().startswith("final answer:"):
            answer_text = re.split(r"final answer:", answer_text, maxsplit=1, flags=re.IGNORECASE)[-1].strip()

        # Disclaimer Enforcement
        normalized_answer_for_disclaimer_check = ' '.join(answer_text.lower().split())
        normalized_disclaimer_en = ' '.join(DISCLAIMER_EN.lower().split())
        normalized_disclaimer_ms = ' '.join(DISCLAIMER_MS.lower().split())

        is_en_disclaimer_at_end = normalized_answer_for_disclaimer_check.endswith(normalized_disclaimer_en)
        is_ms_disclaimer_at_end = normalized_answer_for_disclaimer_check.endswith(normalized_disclaimer_ms)

        if not (is_en_disclaimer_at_end or is_ms_disclaimer_at_end):
            print("Warning: LLM did not include the disclaimer or not at the very end. Appending a default disclaimer.")
            disclaimer_to_append = DISCLAIMER_EN # Default
            # Simple heuristic for language of query
            simple_malay_check = any(k in original_user_query.lower() for k in ["apa", "makna", "sila", "bagaimana", "kenapa", "terangkan", "maksud", "berikan", "jelaskan"])
            if simple_malay_check:
                disclaimer_to_append = DISCLAIMER_MS
            
            # Attempt to remove any partial/incorrect disclaimers before appending
            # This is a simplified version of the user's complex removal logic
            answer_text_lower_for_cleaning = answer_text.lower()
            if normalized_disclaimer_en[:30] in answer_text_lower_for_cleaning and disclaimer_to_append == DISCLAIMER_MS:
                 idx_en = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_en[:20])
                 if idx_en != -1: answer_text = answer_text[:idx_en].strip()
            elif normalized_disclaimer_ms[:30] in answer_text_lower_for_cleaning and disclaimer_to_append == DISCLAIMER_EN:
                 idx_ms = answer_text_lower_for_cleaning.rfind(normalized_disclaimer_ms[:20])
                 if idx_ms != -1: answer_text = answer_text[:idx_ms].strip()


            if answer_text and not answer_text.endswith(("\n\n", "\n")):
                answer_text += "\n\n"
            elif answer_text and not answer_text.endswith("\n\n"): # only one newline
                answer_text += "\n"
            answer_text += disclaimer_to_append
        
        print(f"LLM Formatted Answer (first 200 chars): {answer_text[:200]}...")
        return answer_text, retrieved_documents

    except Exception as e:
        print(f"Error during structured answer generation with LLM: {e}")
        disclaimer_to_append_on_error = DISCLAIMER_EN
        simple_malay_check_on_error = any(k in original_user_query.lower() for k in ["apa", "makna", "sila", "bagaimana", "kenapa", "terangkan", "maksud"])
        if simple_malay_check_on_error:
            disclaimer_to_append_on_error = DISCLAIMER_MS
        return f"An error occurred while generating the structured answer.\n\n{disclaimer_to_append_on_error}", retrieved_documents

