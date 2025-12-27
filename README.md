ChromaDB automatically performs cosine-similarity search when retrieving documents. We don’t calculate it manually. However, ChromaDB does not return the cosine similarity value directly — instead, it returns the cosine distance (called the score). Since cosine distance is defined as 1 − cosine_similarity, we convert it back to similarity in our code using similarity = 1 - score. This gives us the actual cosine similarity value, where higher values mean more relevant results.

1. Create the venv 
2. install all requirements
3. make ur own api key via google ai studio 
4. make .env file and put your api key there 
5. GOOGLE_API_KEY=2jehgg32_33w
6. put yout api key in .env file            
             # GFG_RAG_project
Add your PDF files in data folder and make sure they have text also check with adjusting different score_threshold value .
check for bugs
#  IMP create separate branches and dont push into the main branch 
