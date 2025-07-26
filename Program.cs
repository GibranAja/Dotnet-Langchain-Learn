using LangChain.Providers.OpenAI.Predefined;
using static LangChain.Chains.Chain;
using LangChain.Chains.StackableChains.Agents.Tools.BuiltIn;
using LangChain.Memory;
using LangChain.Providers.OpenAI;
using LangChain.Splitters.Text;
using LangChain.DocumentLoaders;
using LangChain.Databases.Sqlite;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using LangChain.Providers;

var OpenAIKey = "OpenAIKey";
var GoogleSearchKey = "GoogleSearchKey";
var GoogleCx = "GoogleCxKey";

async Task RagDemo()
{
    // prepare OpenAI embedding model
    var provider = new OpenAiProvider(apiKey:
        OpenAIKey);
    var embeddingModel = new TextEmbeddingV3SmallModel(provider);
    var llm = new OpenAiLatestFastChatModel(provider);

    using var vectorDatabase = new SqLiteVectorDatabase("vectors.db");
    var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>(
        embeddingModel,
        dimensions: 1536, // Should be 1536 for TextEmbeddingV3SmallModel
                          // First, specify the source to index.
        dataSource: DataSource.FromPath("E:\\Sekolah Skenik\\Design\\File PDF\\Quantum-Computing.pdf"),
        collectionName: "quantum",
        // Second, configure how to extract chunks from the bigger document.
        textSplitter: new RecursiveCharacterTextSplitter(
            chunkSize: 500, // To pick the chunk size, estimate how much information would be required to capture most passages you'd like to ask questions about.  Too many characters makes it difficult to capture semantic meaning, and too few characters means you are more likely to split up important points that are related. In general, 200-500 characters is good for stories without complex sequences of actions.
            chunkOverlap: 200)); // To pick the chunk overlap you need to estimate the size of the smallest piece of information. It may happen that one chunk ends with `Ron's hair` and the other one starts with `is red`.In this case, an embedding would miss important context, and not be generated propperly. With overlap the end of the first chunk will appear in the begining of the other, eliminating the problem.

    string promptText =
        @"Use the following pieces of context to answer the question at the end. If the answer is not in context then just say that you don't know, don't try to make up an answer. Keep the answer as short as possible.

{context}

Question: {question}
Helpful Answer:";
    while (true)
    {
        Console.Write("Pertanyaan: ");
        var input = Console.ReadLine() ?? string.Empty;
        if (input == "exit")
            break;


        var chain =
        Set(input, outputKey: "question")     // set the question
        | RetrieveDocuments(
            vectorCollection,
            embeddingModel,
            inputKey: "question",
            outputKey: "documents",
            amount: 5)                                                      // take 5 most similar documents
        | StuffDocuments(inputKey: "documents", outputKey: "context")       // combine documents together and put them into context
        | Template(promptText)                                              // replace context and question in the prompt with their values
        | LLM(llm);                                                         // send the result to the language model

        // get chain result
        var res = await chain.RunAsync("text");

        Console.Write("Jawaban: ");
        Console.WriteLine(res);
    }

    Console.WriteLine("bye!!");
}

async Task DemoChat()
{

    var model = new OpenAiLatestFastChatModel(OpenAIKey);

    // create simple template for conversation for AI to know what piece of text it is looking at
    var template =
        @"Kamu adalah seorang pelawak yang selalu menjawab dengan lucu menggunakan bahasa indonesia seperti lawakannya sule, yaitu pelawak terkenal di indonesia, gunakan bahasa gaul jaksel.
{history}
Iban: {input}
AI:";


    // To have a conversation thar remembers previous messages we need to use memory.
    // For memory to work properly we need to specify AI and Human prefixes.
    // Since in our template we have "AI:" and "Human:" we need to specify them here. Pay attention to spaces after prefixes.
    var conversationBufferMemory = new ConversationBufferMemory(new ChatMessageHistory());// TODO: Review { AiPrefix = "AI: ", HumanPrefix = "Human: "};

    // build chain. Notice that we don't set input key here. It will be set in the loop
    var chain =
        // load history. at first it will be empty, but UpdateMemory will update it every iteration
        LoadMemory(conversationBufferMemory, outputKey: "history")
        | Template(template)
        | LLM(model)
        // update memory with new request from Human and response from AI
        | UpdateMemory(conversationBufferMemory, requestKey: "input", responseKey: "text");

    // run an endless loop of conversation
    while (true)
    {
        Console.Write("Iban: ");
        var input = Console.ReadLine() ?? string.Empty;
        if (input == "exit")
            break;

        // build a new chain using previous chain but with new input every time
        var chatChain = Set(input, "input")
                        | chain;

        var res = await chatChain.RunAsync("text");


        Console.Write(" ");
        Console.WriteLine(res);
    }
}

async void DemoHelloWorldAI()
{

    var model = new OpenAiLatestFastChatModel(OpenAIKey);
    var chain =
        Set("Hello!")
        | LLM(model);

    Console.WriteLine(await chain.RunAsync("text"));
    Console.ReadKey();
}

async Task DemoSearchAI()
{

    var llm = new OpenAiLatestFastChatModel(OpenAIKey).UseConsoleForDebug();


    var searchTool = new GoogleCustomSearchTool(key: GoogleSearchKey, cx: GoogleCx, resultsLimit: 1);

    while (true)
    {
        Console.Write("Tanya: ");
        var input = Console.ReadLine() ?? string.Empty;
        if (input == "exit")
            break;
        var chain =
       Set(input)
       | ReActAgentExecutor(llm) // does the magic
           .UseTool(searchTool); // add the google search tool

        await chain.RunAsync();
        Console.WriteLine();
    }

}

//await DemoChat();
//await RagDemo();
await DemoSearchAI();