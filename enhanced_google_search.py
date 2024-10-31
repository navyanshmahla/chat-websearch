import requests
from trafilatura import fetch_url, extract
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import time
import serpapi
import string

client = OpenAI(api_key='OPENAI_API_KEY')

chunk_to_url_mapping = {}

class GoogleSearchGPT:
    def __init__(self, serpapi_key, openai_api_key):
        self.serp_api_key = serpapi_key

    def google_search(self, query):
        params = {
            'engine':'google',
            'q': query,
            "api_key": self.serp_api_key
        }
        search = serpapi.search(params)
        results = search.as_dict()
        organic = results['organic_results']
        gsearch_items = []
        for item in organic:
            gsearch_items.append({
                "title" : item['title'],
                "url" : item['link'],
                "snippet" : item['snippet']
            })
        if len(gsearch_items)==0:
            print(f"No items fetched from the Google Search using SerpAPI")
            return []
        
        return gsearch_items

    @staticmethod
    def extract_between_dots(text):
        texts = text.split("...")
        clean_texts = [x.strip(string.punctuation + string.whitespace) for x in texts]
        gimme = ' '.join(clean_texts)
        return gimme

    def extract_pages(self, retrieved_items):
        complete_text = []
        #extractor = extractors.KeepEverythingExtractor()
        
        for ritem in retrieved_items:
            print(f"Processing URL: {ritem['url']}")
            try:
                # response = requests.get(ritem['url'])
                # response.raise_for_status()  
                # html_content = response.text
                
                # extracted_content = extractor.get_content(html_content)
                print('here we go')
                downloaded = fetch_url(ritem['url'])  
                
                article_text = extract(downloaded)
                if article_text is not None:
                    print("Yayyyy!!")
                    complete_text.append({"url": ritem['url'], "article_text": article_text, "snippet": ritem['snippet']})
                    #print(f"Complete text!!@!@!@!@!@@: {complete_text[-1]['article_text']}\n")
                else:
                    print(f"No content extracted from: {ritem['url']}")
                
                
                
            
            except requests.RequestException as req_err:
                print(f"Request error for URL {ritem['url']}: {req_err}")
            except Exception as ex:
                print(f"Extraction error for URL {ritem['url']}: {ex}")

        return complete_text


    @staticmethod
    def remove_dates(text):
        date_pattern = r'^\s*\w+\s+\d{1,2},\s+\d{4}\s*'
        cleaned_text = re.sub(date_pattern, '', text)
        
        # Return the cleaned text
        return cleaned_text

    @staticmethod
    def extract_context(big_string, substring, word_limit=100):
        # Split the text into words
        print(f"Length of the big string!! : {len(big_string)}", flush=True)
        words = big_string.split()
        
        # Find where the substring appears in the text
        index = big_string.find(substring)
        print(f"INdexxxx: {index}")
        if index == -1:
            return substring
        
        # Find which word contains our substring
        current_position = 0
        word_position = 0
        
        # Loop through words to find which word contains our substring
        for i, word in enumerate(words):
            if current_position <= index < current_position + len(word):
                word_position = i
                break
            current_position += len(word) + 1  # +1 for the space after each word
        
        # Calculate the context window
        start_index = max(0, word_position - word_limit)
        end_index = min(len(words), word_position + word_limit)
        
        # Get the words within our context window
        context_words = words[start_index:end_index]
        
        return ' '.join(context_words)


    def parse_prompts(self, query, context_list):
        retrieval_contexts = [context for context in context_list if context.fetch_type == "retrieval"]
        snippet_search_contexts = [context for context in context_list if context.fetch_type == "snippet_search"]

        retrieval_contexts = sorted(retrieval_contexts, key=lambda x: x.score, reverse=True)
        
        formatted_contexts = []
        for i, context in enumerate(retrieval_contexts):
            text = context.text
            print(f"Chunk for this is : {context}")
            url = context.url
            score = context.score
            context_string = (
                f"URL of the source webpage: {url} "
                f"Score given by the retriever to the context chunk: {score} "
                f"Context chunk text: {text}"
            )
            formatted_contexts.append(f"[{i + 1}] {context_string}")

        snippet_index_offset = len(formatted_contexts)
        for i, context in enumerate(snippet_search_contexts):
            text = context.text
            url = context.url
            context_string = (
                f"URL of the source webpage: {url} "
                f"Score not valid since it's not a chunk but a complete piece of text. "
                f"Context text: {text}"
            )
            formatted_contexts.append(f"[{snippet_index_offset + i + 1}] {context_string}")
        
        context = "\n".join(formatted_contexts)

        with open('user_prompt.txt') as file:
            user_template = file.read()
        user_prompt = user_template.replace('$QUERY', query).replace('$CONTEXTS', context)

        with open('system_prompt.txt') as file:
            system_prompt = file.read()
        
        print(user_prompt)
        return user_prompt, system_prompt


    def get_response(self, prompt, system_prompt):
        try:
            response = client.chat.completions.create(model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"

    def execute2(self, query):
        search_results = self.google_search(query)
        extracted_texts = self.extract_pages(search_results)
        # print(extracted_texts[0]['article_text'])
        #print(extracted_texts)

        snippets = [self.remove_dates(self.extract_between_dots(item['snippet'])) for item in search_results]
        contexts = []
        for extracted_text, snippet in zip(extracted_texts, snippets):
            #print(f"Extracted Text $$$$$: {extracted_text['article_text']}\n")
            #print(f"Type of extracted text: {type(extracted_text['article_text'])}")
            print(f"Snippet is actualy: {snippet}\n", flush=True)
            cont = GoogleSearchGPT.extract_context(extracted_text['article_text'], snippet)
            #print(f"Contextttt: {cont}\n")
            contexts.append(cont)
        # contexts = [self.extract_context(extracted_text['article_text'], snippet) for extracted_text, snippet in zip(extracted_texts, snippets)]
        

        user_prompt, system_prompt = self.parse_prompts(query, contexts)
        return self.get_response(user_prompt, system_prompt)


@dataclass
class SearchResult:
    url: str
    text: str
    fetch_type : str # allowed values : ['snippet_search', 'retrieval']
    score: float = 0.0
    
class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
            
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
            
        return chunks

class BGEM3Reranker:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_scores(self, query: str, texts: List[str]) -> List[float]:
        
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        text_embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        similarities = (query_embedding @ text_embeddings.T).tolist()
        return similarities if isinstance(similarities, list) else [similarities]

class EnhancedGoogleSearchGPT(GoogleSearchGPT):
    def __init__(self, serpapi_key: str, openai_api_key: str):
        super().__init__(serpapi_key, openai_api_key)
        self.chunker = TextChunker()
        self.reranker = BGEM3Reranker()
        self.use_chunking = False 
        self.use_retriever = False

    def extract_context(self, big_string, substring, indices, upper_limit=100, lower_limit=200):
    
        #print(f"Length of the big string!! : {len(big_string)}", flush=True))
        #print(type(big_string))

        if not isinstance(big_string, str):
            #print(big_string)
            raise TypeError("'big_string' parameter must be a string")
        words = big_string.split()
        contexts = []

        for index in indices:
            current_position = 0
            word_position = 0
        
            for i, word in enumerate(words):
                if current_position <= index < current_position + len(word):
                    word_position = i
                    break
                current_position += len(word) + 1  
            
            start_index = max(0, word_position - upper_limit)
            end_index = min(len(words), word_position + lower_limit)
            
            context_words = words[start_index:end_index]
            contexts.append(' '.join(context_words))
        
        return contexts

    
    def extract_between_dots(self, text):
        texts = text.split("...")
        clean_texts = [x.strip(string.punctuation + string.whitespace) for x in texts]
        gimme = ' '.join(clean_texts)
        return gimme

    def process_search_results(self, extracted_texts: List[Dict[str, str]], query: str) -> List[SearchResult]:
        processed_results = []

        for item in extracted_texts:
            text = item['article_text']
            if not isinstance(text, str):
                print(f"text isn't a string but its: {type(text)}")
            snippet = item['snippet']
            print(type(snippet))
            cleaned_snippet = self.extract_between_dots(snippet)

            #index = text.find(cleaned_snippet)
            indices = []

            start = 0

            while start < len(text):
                index = text.find(cleaned_snippet, start)
                if index == -1:
                    break
                indices.append(index)
                start = index + 1  # move past the last found index

            if len(indices)==0:
                print(f"No instance of snippet found in this URL, proceeding with chunking and retrieval!")
                self.use_retriever = True
            
            if self.use_retriever:
                if self.use_chunking:
                    start = time.time()
                    chunks = self.chunker.create_chunks(text)

                    # for chunk in chunks:]
                    #     chunk_to_url_mapping[chunk] = item['url']

                    scores = self.reranker.compute_scores(query, chunks)

                    scored_chunks = list(zip(scores, chunks))
                    scored_chunks.sort(key=lambda x: x[1], reverse=True)

                    # configuring the max number of chunks that come from a document
                    top_chunks = 3
                    chunks = chunks[:top_chunks]
                    
                    top_scored_chunks = scored_chunks[:top_chunks]
                    top_scores = []

                    for score, chunk in top_scored_chunks:
                        chunk_to_url_mapping[chunk] = item['url']
                        top_scores.append(score)

                    for score,chunk in top_scored_chunks:
                        processed_results.append(SearchResult(
                            url = item['url'],
                            text = chunk,
                            score = score,
                            fetch_type = "retrieval"
                        ))
                
                    end = time.time()
                    print(f"TIME TAKEN MANNN!!! {end-start}\n", flush=True)
                else:
                    start = time.time()
                    score = self.reranker.compute_scores(query, [text])[0]
                    end = time.time()
                    print(f"TIME TAKEN MANNN!!! {end-start}\n", flush=True)
                    processed_results.append(SearchResult(
                        url=item['url'],
                        text=text,
                        score=score,
                        fetch_type = "retrieval"
                    ))
            
            else: 
                contexts = self.extract_context(text, snippet, indices)
                for context in contexts:
                    processed_results.append(SearchResult(
                        url = item['url'],
                        text = context,
                        score = 0.0,
                        fetch_type = "snippet_search"
                    ))
                
        for result in processed_results:
            if type(result.score) == str:
                print("AAAAAHHHHH!!!")
                print(result.score)       
                print(result.fetch_type)

        processed_results.sort(key=lambda x: x.score, reverse=True)
        return processed_results
        
        

    def execute(self, query: str) -> str:
        search_results = self.google_search(query)
        extracted_texts = self.extract_pages(search_results)

        for item in search_results:
            print(f"SNIPPETS ARE: {item['snippet']}", flush=True)
        

        ranked_results = self.process_search_results(extracted_texts, query)
        
        
        top_k = 8  # only accounts for retrieval, the snippet_search contexts are all included.
        top_k_snippet_docs = 8
        contexts = []
        count = 0
        count_snippet = 0

        for result in ranked_results:
            if result.fetch_type == "snippet_search":
                count_snippet+=1
                contexts.append(result)
                if count_snippet==top_k_snippet_docs:
                    break

            elif result.fetch_type == "retrieval":
                count+=1
                contexts.append(result)
                if count==top_k:
                    break
        
        #contexts = [result.text for result in ranked_results[:top_k]]
        
        
        user_prompt, system_prompt = self.parse_prompts(query, contexts)
        return self.get_response(user_prompt, system_prompt)

# Example usage
if __name__ == "__main__":
    SERPAPI_KEY = 'SERP_API_KEY'
    OPENAI_API_KEY = 'OPENAI_API_KEY'

    gpt_instance = EnhancedGoogleSearchGPT(SERPAPI_KEY, OPENAI_API_KEY)
    
    gpt_instance.use_chunking = True
    
    query = "how can I increase my verticle jump while dunking in basketball"
    response = gpt_instance.execute(query)
    print("\n###FINAL RESPONSE\n")
    print(response)

## TO DO: Write a scraping module from scratch that even works for dynamic webpages and better extraction of text content. Maybe even add functionality for images and figures on the webpages