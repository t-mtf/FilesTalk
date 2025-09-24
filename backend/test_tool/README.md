The test tool is designed to automate the evaluation of inflation clause-related information from contract documents. It processes data in two ways:

1. **Comparison with Reference Dataset**: The tool analyzes an Excel file containing extracted data from contract documents and compares the results with a reference test dataset to ensure the results are accurate and reliable.

2. **Similarity Assessment**: It compares two Excel extraction files to determine if each value is "identical," "nearly identical," or "distinct."

The tool operates in several key steps:

1. **Data Synthesis**: The tool synthesizes results at the contract level, aggregating information from various documents within each contract. This involves using a language model to analyze and synthesize responses related to a contract number.

2. **Comparison with Reference Dataset**: After synthesizing the results, the tool compares them against a reference test dataset. This comparison is facilitated by a function that uses a language model to determine the similarity between the extracted results and the reference data, categorizing them as "identical," "nearly identical," or "distinct."

3. **Performance Metrics Calculation**: The tool calculates performance metrics for the extracted data, treating the identification of inflation clauses as a classification problem. It computes accuracy, precision, and recall for the `inflation_found` column, with a focus on recall due to its importance in identifying relevant clauses. Additionally, it calculates accuracy for other columns, such as inflation type, effective date, notice period, and periodicity.

Overall, the tool provides a systematic approach to extracting, synthesizing, and evaluating inflation clause information from contracts, leveraging advanced language models to ensure the results are accurate and reliable.
