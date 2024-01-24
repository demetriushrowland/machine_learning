curl https://api.openai.com/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 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" \
-d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Answer the following question with A, B, C, or D. What is the minimum of the following polynomial: 1.4x^2 + -2.3x + -2.5 ?(A) 1.22 (B) 0.5 (C) 0.82 (D) 1.72 "}],
     "temperature": 0.7
   }' > response.txt