graph TD
    A[Start] --> B[Initialize App]
    B --> C[Load API Keys & Configure Page]
    C --> D[Initialize RAG System]
    D --> E[Display UI Components]

    E --> F{User Input}
    F --> G1[Campaign Details]
    F --> G2[Target Market]
    F --> G3[Advanced Settings]

    G1 --> H[Generate Button Clicked]
    G2 --> H
    G3 --> H

    H --> I[Validate Inputs]
    I -- Invalid --> E
    I -- Valid --> J[Check RAG Option]

    J -- RAG Enabled --> K[Retrieve Context]
    J -- RAG Disabled --> W[Check Web Search]
    K --> W

    W -- Web Search Enabled --> X[Fetch Search Results]
    W -- Web Search Disabled --> L[Generate Content]
    X --> L

    L --> M{Content Type}
    M --> N1[Social Media]
    M --> N2[Email]
    M --> N3[Marketing]
    M --> N4[Text]

    N1 --> O[Display Content]
    N2 --> O
    N3 --> O
    N4 --> O

    O --> P{Generate Image?}
    P -- Yes --> Q[Generate & Display Image]
    P -- No --> R[Save Options]
    Q --> R

    R --> S[End]