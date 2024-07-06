##### From https://sausheong.com/programming-with-ai-vision-llms-e4d1107c0be1

## OpenAI GPT 4o
func gpt(model string, filepath string, query string) (string, error) {
 c := context.Background()
 llm, err := openai.New(openai.WithModel(model))
 if err != nil {
  return "", err
 }

 dataurl, err := imageurl(filepath)
 if err != nil {
  fmt.Println("Error reading image file:", err)
  return "", err
 }

 resp, err := llm.GenerateContent(
  c, []llms.MessageContent{
   {
    Role: llms.ChatMessageTypeHuman,
    Parts: []llms.ContentPart{
     llms.ImageURLPart(dataurl),
     llms.TextPart(query),
    },
   },
  },
 )
 if err != nil {
  return "", err
 }
 return resp.Choices[0].Content, nil
}

func imageurl(filepath string) (string, error) {
 imageData, err := os.ReadFile(filepath)
 if err != nil {
  fmt.Println("Error reading image file:", err)
  return "", err
 }
 contentType := http.DetectContentType(imageData[:512])
 return "data:" + contentType + ";base64," + 
          base64.StdEncoding.EncodeToString(imageData), nil
}

func show(filepath string) {
 imageData, err := os.ReadFile(filepath)
 if err != nil {
  fmt.Println("Error reading image file:", err)
  return
 }
}


## Google Gemini 1.5 Pro
func gemini(model string, filepath string, query string) (string, error) {
 c := context.Background()
 llm, err := googleai.New(c,
  googleai.WithDefaultModel(model),
  googleai.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
 if err != nil {
  return "", err
 }

 imageData, err := os.ReadFile(filepath)
 if err != nil {
  fmt.Println("Error reading image file:", err)
  return "", err
 }
 contentType := http.DetectContentType(imageData[:512])

 resp, err := llm.GenerateContent(
  c, []llms.MessageContent{
   {
    Role: llms.ChatMessageTypeHuman,
    Parts: []llms.ContentPart{
     llms.BinaryPart(contentType, imageData),
     llms.TextPart(query),
    },
   },
  },
 )
 if err != nil {
  return "", err
 }
 return resp.Choices[0].Content, nil
}

## Llava 1.6
func local(model string, filepath string, query string) (string, error) {
 c := context.Background()
 llm, err := ollama.New(ollama.WithModel(model))
 if err != nil {
  return "", err
 }

 imageData, err := os.ReadFile(filepath)
 if err != nil {
  fmt.Println("Error reading image file:", err)
  return "", err
 }
 contentType := http.DetectContentType(imageData[:512])

 resp, err := llm.GenerateContent(
  c, []llms.MessageContent{
   {
    Role: llms.ChatMessageTypeHuman,
    Parts: []llms.ContentPart{
     llms.BinaryPart(contentType, imageData),
     llms.TextPart(query),
    },
   },
  },
 )
 if err != nil {
  return "", err
 }
 return resp.Choices[0].Content, nil
}



