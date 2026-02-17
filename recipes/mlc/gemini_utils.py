from google import genai

def get_context(audio_path, prompt, client):
    myfile = client.files.upload(file=audio_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[prompt, myfile]
    )
    return response

def generate_response(audio_path, prompt, client, config=None, model="gemini-2.5-flash"):
    myfile = client.files.upload(file=audio_path)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, myfile],
        config=config,
    )
    return response, myfile

def generate_response_multi(audio_path, prompt, client, config=None, model="gemini-2.5-flash"):
    files = [ client.files.upload(file=path) for path in audio_path ]

    response = client.models.generate_content(
        model=model,
        contents=[prompt, *files],
        config=config,
    )
    for file in files:
        client.files.delete(name=file.name)
    return response

def construct_prompt(question, answer):
    #prompt = f"Given the supplied audio, the supplied question about the audio, and the ground-truth answer to the question, please provide some short, concise, but useful context that would help support the answer with regard to the question. Provide a few short sentences.\nQuestion: {question}\nCorrect Answer: {answer}"
    #prompt = f"Given the supplied audio, and the supplied question about the audio, please provide a description of the audio that is relevant to what the question is asking about. The provided description will subsequently be used to evaluate candidate open-ended answers to the question, so if you deem it necessary, provide also some brief, for example cultural context that would provide a better basis for evaluating those candidate answers.\nQuestion: {question}"
    prompt = f"Given the supplied audio, and the supplied question about the audio, please provide a description of the audio that is relevant to what the question is asking about. The provided description will subsequently be used to evaluate candidate open-ended answers to the question.\nQuestion: {question}"
    prompt = f"Provide a description of the following audio. The provided description will subsequently be used to evaluate candidate open-ended answers to potential questions about the audio, so try to capture all the relevant aspects in reasonable detail."
    prompt = f"Provide a description of the following audio. The provided description will subsequently be used to evaluate candidate open-ended answers to potential questions about the audio, so try to capture all the relevant aspects in reasonable detail. Additionally, given the following example question and ground truth answer, try to also tie the description to the question-answer pair.\nQuestion: {question}\nGround Truth Answer: {answer}"
    return prompt

def construct_prompt_open_ended(question):
    #prompt = f"Given the supplied audio and question, provide a short, concise answer.\nQuestion: {question}"
    #prompt = f"Given the supplied audio and question, provide a concise answer.\nQuestion: {question}"
    #prompt = f"Given the supplied audio and question, provide some brief rationale and then include the answer like so: 'Answer: %answer%'.\nQuestion: {question}"
    prompt = f"{question}"
    #prompt = f"Given the supplied audio and the supplied question about the audio, please provide a few sentences as a description of the audio. The description will later be used to evaluate candidate answers to the supplied question, so try to also tie the description to the question.\nQuestion: {question}"
    return prompt

def load_api_keys(path='./api_keys'):
    with open(path, 'r') as f:
        api_keys = [line.split("#")[0].strip().replace('"', "") for line in f if line.strip()]
    return api_keys

class ClientCycler:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.index = 0

    def get_next_client(self):
        try:
            client = genai.Client(api_key=self.api_keys[self.index])
        except IndexError:
            print("No more API keys available.")
            return None

        self.index = self.index + 1

        i = 0
        for f in client.files.list():
            if i > 100:
                break
            client.files.delete(name=f.name)
            i += 1
        
        return client
