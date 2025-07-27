using LangChain.Providers.OpenAI.Predefined;
using static LangChain.Chains.Chain;
using LangChain.Chains.StackableChains.Agents.Tools.BuiltIn;
using LangChain.Memory;
using LangChain.Providers.Automatic1111;
using LangChain.Providers.OpenAI;
using LangChain.Splitters.Text;
using LangChain.DocumentLoaders;
using LangChain.Databases.Sqlite;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using LangChain.Providers;
using OpenAI.Images;
using LangChain.Chains.StackableChains.Agents.Crew.Tools;
using LangChain.Chains.StackableChains.Agents.Crew;

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

async Task DemoImageGeneration()
{
    var provider = new OpenAiProvider(apiKey: OpenAIKey);
    var llm = new OpenAiLatestFastChatModel(provider);


    var sdmodel = new Automatic1111Model
    {
        Settings = new Automatic1111ModelSettings
        {
            NegativePrompt = "bad quality, blured, watermark, text, naked, nsfw",
            Seed = 42, // for results repeatability
            CfgScale = 6.0f,
            Width = 512,
            Height = 768,
        },
    };


    var template =
        @"[INST]Transcript of a dialog, where the User interacts with an Assistant named Stablediffy. Stablediffy knows much about prompt engineering for stable diffusion (an open-source image generation software). The User asks Stablediffy about prompts for stable diffusion Image Generation. 

Possible keywords for stable diffusion: ""cinematic, colorful background, concept art, dramatic lighting, high detail, highly detailed, hyper realistic, intricate, intricate sharp details, octane render, smooth, studio lighting, trending on artstation, landscape, scenery, cityscape, underwater, salt flat, tundra, jungle, desert mountain, ocean, beach, lake, waterfall, ripples, swirl, waves, avenue, horizon, pasture, plateau, garden, fields, floating island, forest, cloud forest, grasslands, flower field, flower ocean, volcano, cliff, snowy mountain
city, cityscape, street, downtown""
[/INST]
-- Transcript --

USER: suggest a prompt for a young girl from Swiss sitting by the window with headphones on
ASSISTANT: gorgeous young Swiss girl sitting by window with headphones on, wearing white bra with translucent shirt over, soft lips, beach blonde hair, octane render, unreal engine, photograph, realistic skin texture, photorealistic, hyper realism, highly detailed, 85mm portrait photography, award winning, hard rim lighting photography

USER: suggest a prompt for an mysterious city
ASSISTANT: Mysterious city, cityscape, urban, downtown, street, noir style, cinematic lightning, dramatic lightning, intricate, sharp details, octane render, unreal engine, highly detailed, night scene, dark lighting, gritty atmosphere

USER: suggest a prompt for a high quality render of a car in 1950
ASSISTANT: Car in 1950, highly detailed, classic car, 1950's, highly detailed, dramatic lightning, cinematic lightning, unreal engine

USER:suggest a prompt for {value}
ASSISTANT:";


    var chain = Set("a cute girl cosplaying a cat")                                     // describe a desired image in simple words
                | Template(template, outputKey: "prompt")                               // insert our description into the template
                | LLM(llm, inputKey: "prompt", outputKey: "image_prompt")           // ask ollama to generate a prompt for stable diffusion
                | GenerateImage(sdmodel, inputKey: "image_prompt", outputKey: "image")  // generate an image using stable diffusion
                | SaveIntoFile("image.png", inputKey: "image");                     // save the image into a file

    // run the chain
    await chain.RunAsync();
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

async Task DemoTool()
{

    var provider = new OpenAiProvider(apiKey: OpenAIKey);
    var llm = new OpenAiLatestFastChatModel(provider);
    var model = new OpenAiChatModel(provider, id: "gpt-4o-mini").UseConsoleForDebug();


    var imageTool = new CrewAgentToolLambda("create_image", "create image from prompt", prompt =>
    {
        ImageClient client = new("dall-e-3", OpenAIKey);
        
        ImageGenerationOptions options = new()
        {
            Quality = GeneratedImageQuality.High,
            Size = GeneratedImageSize.W1024xH1024,
            Style = GeneratedImageStyle.Vivid,
            ResponseFormat = GeneratedImageFormat.Uri
        };
        GeneratedImage image = client.GenerateImage(prompt, options);
        //BinaryData bytes = image.ImageBytes;
        return Task.FromResult($"Image telah berhasil di generate, ini image urlnya : {image.ImageUri}");
    });


    // the actual agent who does the job
    var desainer = new CrewAgent(model, "desianer", "generate image from user description",
        "you use create_image tool to create image from user description");


    desainer.AddTools(new[] { imageTool });

    // controls agents
    var manager = new CrewAgent(model, "manager", "assign task to one of your co-workers and return the result");


    var chain =
        Set("Buatkan saya gambar sapi yang sedang terbang di angkasa luas")
        | Crew(new[] { manager ,desainer }, manager);

    var res = await chain.RunAsync("text");
    Console.WriteLine(res);

    //--- ---

   
}

await DemoTool();
//await DemoChat();
//await RagDemo();
//await DemoSearchAI();