# Visual Diagrams in Mermaid Format

These diagrams can be rendered using Mermaid (GitHub, VS Code, or online: mermaid.live)

---

## 1. GANTT CHART (Mermaid)

```mermaid
gantt
    title Plant Disease Detection Project Timeline
    dateFormat YYYY-MM-DD
    section Phase 1: Planning
    Project Proposal          :a1, 2024-01-01, 14d
    Literature Review         :a2, 2024-01-01, 14d
    Dataset Acquisition       :a3, 2024-01-08, 7d
    System Design             :a4, 2024-01-15, 7d
    
    section Phase 2: Setup
    Environment Setup         :b1, 2024-01-15, 7d
    Install Python/TensorFlow :b2, 2024-01-15, 7d
    Install Flask/MySQL       :b3, 2024-01-15, 7d
    
    section Phase 3: Data Prep
    Dataset Exploration       :c1, 2024-01-22, 7d
    Data Preprocessing        :c2, 2024-01-22, 7d
    Data Split                :c3, 2024-01-29, 7d
    Data Augmentation         :c4, 2024-01-29, 7d
    
    section Phase 4: Model Dev
    CNN Design                :d1, 2024-02-05, 7d
    CNN Implementation        :d2, 2024-02-05, 14d
    CNN Training              :d3, 2024-02-12, 14d
    MobileNet Design          :d4, 2024-02-05, 7d
    MobileNet Implementation  :d5, 2024-02-12, 14d
    MobileNet Training        :d6, 2024-02-19, 14d
    
    section Phase 5: Evaluation
    CNN Testing               :e1, 2024-03-04, 7d
    MobileNet Testing         :e2, 2024-03-11, 7d
    Performance Comparison    :e3, 2024-03-11, 7d
    Model Optimization        :e4, 2024-03-18, 7d
    
    section Phase 6: Web App
    Database Implementation   :f1, 2024-03-18, 7d
    Backend Flask API         :f2, 2024-03-18, 14d
    Frontend Development      :f3, 2024-03-18, 14d
    Authentication            :f4, 2024-03-25, 7d
    Image Upload Feature      :f5, 2024-04-01, 7d
    Disease Info Database     :f6, 2024-04-01, 7d
    
    section Phase 7: Integration
    Model-Flask Integration   :g1, 2024-04-01, 7d
    End-to-End Testing        :g2, 2024-04-08, 7d
    Bug Fixes                 :g3, 2024-04-08, 7d
    User Acceptance Testing   :g4, 2024-04-08, 7d
    
    section Phase 8: Deployment
    Documentation             :h1, 2024-04-15, 7d
    Report Writing            :h2, 2024-04-15, 7d
    Deployment Setup          :h3, 2024-04-22, 7d
    Final Presentation        :h4, 2024-04-22, 7d
```

---

## 2. STATE DIAGRAM - User Authentication (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> Welcome: Start
    
    Welcome --> Register: Click Register
    Welcome --> Login: Click Login
    
    Register --> ValidateRegistration: Submit Form
    ValidateRegistration --> ErrorMessage: Email Exists
    ValidateRegistration --> SuccessRedirect: New Email
    
    Login --> ValidateCredentials: Submit Form
    ValidateCredentials --> InvalidLogin: Wrong Credentials
    ValidateCredentials --> ValidLogin: Correct Credentials
    
    SuccessRedirect --> HomePage: Success
    ValidLogin --> HomePage: Success
    ErrorMessage --> Register: Try Again
    InvalidLogin --> Login: Try Again
    
    HomePage --> [*]: Logout
```

---

## 3. STATE DIAGRAM - Prediction Workflow (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> UserUploads: Start
    
    UserUploads --> ValidateImage: Upload Complete
    
    ValidateImage --> InvalidFormat: Invalid Type
    ValidateImage --> ValidProcess: Valid Format
    
    InvalidFormat --> [*]: Show Error
    
    ValidProcess --> PreprocessImage: Valid
    PreprocessImage --> LoadModel: Ready
    LoadModel --> ExtractFeatures: Loaded
    ExtractFeatures --> ClassifyDisease: Features Ready
    ClassifyDisease --> GetDiseaseInfo: Class Predicted
    GetDiseaseInfo --> DisplayResults: Info Retrieved
    DisplayResults --> [*]: Complete
    
    note right of PreprocessImage
        Resize to 224x224
        Normalize pixels
    end note
    
    note right of LoadModel
        Load CNN or MobileNet
        Based on user selection
    end note
    
    note right of DisplayResults
        Show disease name,
        confidence score,
        causes & remedies
    end note
```

---

## 4. STATE DIAGRAM - Model Training (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> LoadDataset: Start Training
    
    LoadDataset --> SplitData: Dataset Loaded
    SplitData --> ApplyAugmentation: Splits Ready
    ApplyAugmentation --> ChooseModel: Augmented
    
    ChooseModel --> CNNTrain: Choose CNN
    ChooseModel --> MobileNetTrain: Choose MobileNet
    
    CNNTrain --> CNNValidate: Training Complete
    CNNValidate --> CheckAccCNN: Validated
    CheckAccCNN --> SaveModelCNN: Best Accuracy
    CheckAccCNN --> CNNTrain: Not Best
    
    MobileNetTrain --> MobileNetValidate: Training Complete
    MobileNetValidate --> CheckAccMobile: Validated
    CheckAccMobile --> SaveModelMobile: Best Accuracy
    CheckAccMobile --> MobileNetTrain: Not Best
    
    SaveModelCNN --> [*]
    SaveModelMobile --> [*]
```

---

## 5. PERT CHART (Mermaid)

```mermaid
graph TD
    Start([START]) --> R1[1.1 Requirements]
    Start --> R2[1.2 Literature Review]
    
    R1 --> DS[1.4 System Design]
    R2 --> DS
    
    DS --> ES[2.1 Environment Setup]
    
    ES --> PT[2.2 Python/TensorFlow]
    ES --> FM[2.3 Flask/MySQL]
    
    PT --> DE[3.1 Dataset Exploration]
    FM --> DE
    
    DE --> DP[3.2 Preprocessing]
    DE --> DSplit[3.3 Data Split]
    
    DSplit --> DA[3.4 Augmentation]
    
    DA --> CD[4.1 CNN Design]
    DA --> MD[4.4 MobileNet Design]
    
    CD --> CI[4.2 CNN Implementation]
    CI --> CT[4.3 CNN Training]
    
    MD --> MI[4.5 MobileNet Implementation]
    MI --> MT[4.6 MobileNet Training]
    
    CT --> CTV[5.1 CNN Testing]
    MT --> MTV[5.2 MobileNet Testing]
    
    CTV --> PC[5.3 Performance Comparison]
    MTV --> PC
    
    PC --> MO[5.4 Model Optimization]
    
    MO --> DB[6.1 Database Implementation]
    
    DB --> BE[6.2 Backend Flask]
    DB --> FE[6.3 Frontend HTML]
    
    BE --> Auth[6.4 Authentication]
    FE --> Auth
    
    Auth --> IU[6.5 Image Upload]
    IU --> DI[6.6 Disease Info]
    
    DI --> MFI[7.1 Model-Flask Integration]
    MFI --> E2E[7.2 End-to-End Testing]
    
    E2E --> BF[7.3 Bug Fixes]
    E2E --> UAT[7.4 User Acceptance Testing]
    
    BF --> DOC[8.1 Documentation]
    DOC --> REP[8.2 Report Writing]
    
    REP --> DEP[8.3 Deployment]
    REP --> PRES[8.4 Presentation]
    
    DEP --> End([END])
    PRES --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style MO fill:#87CEEB
    style E2E fill:#DDA0DD
```

---

## 6. SEQUENCE DIAGRAM - Prediction Flow (Mermaid)

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Flask
    participant Model
    participant Database
    
    User->>Browser: Upload Image
    Browser->>Flask: POST /prediction
    Flask->>Flask: Validate & Save Image
    Flask->>Model: Load Trained Model
    Model-->>Flask: Model Ready
    Flask->>Flask: Preprocess Image
    Flask->>Model: Predict Class
    Model-->>Flask: Disease Class + Probability
    Flask->>Database: Get Disease Info
    Database-->>Flask: Causes & Remedies
    Flask->>Flask: Format Results
    Flask-->>Browser: Return JSON
    Browser-->>User: Display Results
```

---

## 7. CLASS DIAGRAM - System Architecture (Mermaid)

```mermaid
classDiagram
    class WebApp {
        +Flask app
        +routes()
        +login()
        +register()
        +upload_image()
        +predict()
    }
    
    class MLModel {
        +CNN model
        +MobileNet model
        +load_model()
        +preprocess()
        +predict()
    }
    
    class Database {
        +MySQL connection
        +users table
        +disease_info table
        +store_user()
        +get_disease_info()
    }
    
    class Preprocessor {
        +resize_image()
        +normalize()
        +augment()
    }
    
    class Frontend {
        +HTML templates
        +CSS styles
        +JavaScript
        +display_results()
    }
    
    WebApp --> MLModel : uses
    WebApp --> Database : queries
    WebApp --> Frontend : renders
    MLModel --> Preprocessor : uses
    MLModel --> Database : retrieves info
```

---

## 8. USE CASE DIAGRAM (Mermaid)

```mermaid
graph TB
    Admin[👤 Admin]
    User[👤 User]
    System[🤖 System]
    
    Admin --> |Manage Users| UC1[Manage User Accounts]
    Admin --> |View Statistics| UC2[View System Stats]
    
    User --> |Register| UC3[Create Account]
    User --> |Login| UC4[Authenticate]
    User --> |Logout| UC5[End Session]
    User --> |Upload| UC6[Upload Leaf Image]
    User --> |View| UC7[View Prediction Results]
    User --> |History| UC8[View Past Predictions]
    
    System --> |Process| UC9[Classify Disease]
    System --> |Store| UC10[Store User Data]
    System --> |Retrieve| UC11[Retrieve Disease Info]
    
    style Admin fill:#FFE4B5
    style User fill:#E0E0E0
    style System fill:#87CEEB
```

---

## How to Use These Diagrams

### Online (Recommended):
1. Go to: https://mermaid.live
2. Copy any diagram code above
3. Paste into the editor
4. Click "PNG" or "SVG" to export

### In VS Code:
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file in VS Code
3. Click Preview (Ctrl+Shift+V)
4. Diagrams will render automatically

### In GitHub:
1. Create a `.md` file in your repository
2. Copy diagram code
3. Push to GitHub
4. GitHub automatically renders Mermaid diagrams!

### In Documentation:
1. Export as PNG/SVG from mermaid.live
2. Insert into Word/PowerPoint documents
3. Use for presentations and reports

---

**All diagrams are ready to use! Just copy and render! 🎨**

