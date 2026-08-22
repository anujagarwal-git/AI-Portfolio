# Prototype chunking comparison — answers only

From `evaluation/eval_result_*.json`. One document (SR 11-7), 11 questions.
Retrieved chunks omitted deliberately — this is for judging the ANSWERS.

## Before you read

The scores shown are **not trustworthy**, which is why you are judging by eye:

- `semantic` retrieval is byte-identical to `recursive` on all 11 questions. It was never a real semantic run. Its answers differ only where generation happened to vary.
- The judge is non-deterministic: sr-03 has identical context AND identical response across runs, scored 0.5 then 1.0.
- The judge is provably wrong on sr-01, where a verbatim quote of SR 11-7 scored faithfulness 0.00.

### Judge each answer on

1. **Correct?** Does it state the requirement accurately?
2. **Complete?** Anything material missing versus the reference?
3. **Padded?** Invented structure, hedging, or bullet lists the source doesn't support?
4. **Actionable?** Could a validator use it as written?

---

## sr-01 — How does SR 11-7 define a model?

### Reference answer (your golden set)

> A model is a quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates. It has three components: an information input component, a processing component, and a reporting component.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 4 chunks, 21,239 chars retrieved</sub>

The definition of a model as stated in SR 11-7 is: "A model refers to a quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates." [SR11-7 §III.1]

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 5,270 chars retrieved</sub>

The definition of a model provided in SR 11-7 is that a model refers to a quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates. [SR11-7 §V.3]

### semantic — same retrieval as `recursive`, answer differs only by generation variance

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 5,270 chars retrieved</sub>

SR 11-7 defines a model as a quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates. [SR11-7 §V.3]

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-02 — What is model risk and what are its two main sources?

### Reference answer (your golden set)

> Model risk is the potential for adverse consequences from decisions based on incorrect or misused model outputs. It arises from two sources: (1) a model may have fundamental errors and produce inaccurate outputs relative to its design objective and intended use, and (2) a model may be used incorrectly or inappropriately.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 4 chunks, 21,239 chars retrieved</sub>

Model risk refers to the potential for adverse consequences from decisions based on incorrect or misused model outputs and reports [Fotnote 3]. The two main sources of model risk are: 

1. The model may have fundamental errors and may produce inaccurate outputs when viewed against the design objective and intended business uses [Fotnote 3]. 
2. The model may be used incorrectly or inappropriately, even if it is fundamentally sound and produces accurate outputs consistent with the design objective [Fotnote 3].

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 3,749 chars retrieved</sub>

Model risk refers to the risk that a model may produce inaccurate or misleading results, which can lead to incorrect decisions or financial losses [III] OVERVIEW OF MODEL RISK MANAGEMENT. The two main sources of model risk are:  
1. **Greater model complexity** and **higher uncertainty about inputs and assumptions** [III] OVERVIEW OF MODEL RISK MANAGEMENT].  
2. **Broader use** and **larger potential impact** of the model [III] OVERVIEW OF MODEL RISK MANAGEMENT].

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 3,749 chars retrieved</sub>

Model risk refers to the risk that a model may produce inaccurate or misleading results, which can lead to incorrect decisions or financial losses [III] OVERVIEW OF MODEL RISK MANAGEMENT. The two main sources of model risk are:  
1. **Greater model complexity** and **higher uncertainty about inputs and assumptions** [III] OVERVIEW OF MODEL RISK MANAGEMENT].  
2. **Broader use** and **larger potential impact** of the model [III] OVERVIEW OF MODEL RISK MANAGEMENT].

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-03 — What is 'effective challenge' in model risk management?

### Reference answer (your golden set)

> Effective challenge is critical analysis by objective, informed parties who can identify model limitations and assumptions and produce appropriate changes. It depends on a combination of incentives, competence, and influence.

### parentdoc

<sub>context_recall 0.5 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 4 chunks, 13,713 chars retrieved</sub>

"Effective challenge" in model risk management refers to the critical analysis by objective, informed parties who can identify model limitations and assumptions and produce appropriate changes. This concept is supported by [VI.5 §VI.5].

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.5 &nbsp;|&nbsp; 5 chunks, 5,931 chars retrieved</sub>

"Effective challenge" in model risk management refers to the critical analysis by objective, informed parties who can identify model limitations and assumptions and produce appropriate changes. This concept depends on a combination of incentives, competence, and influence. [III] OVERVIEW OF MODEL RISK MANAGEMENT

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 5,931 chars retrieved</sub>

"Effective challenge" in model risk management refers to the critical analysis by objective, informed parties who can identify model limitations and assumptions and produce appropriate changes. This concept depends on a combination of incentives, competence, and influence. [III] OVERVIEW OF MODEL RISK MANAGEMENT

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-04 — What are the three core elements of a comprehensive model validation framework?

### Reference answer (your golden set)

> Evaluation of conceptual soundness (including developmental evidence), ongoing monitoring (including process verification and benchmarking), and outcomes analysis (including backtesting).

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 22,673 chars retrieved</sub>

The three core elements of a comprehensive model validation framework are:

1. **Robust Model Development**  
   This involves creating high-quality, accurate, and well-justified models that are grounded in sound assumptions, data, and methodologies. It includes thorough testing, validation, and documentation to ensure that the model is reliable and appropriate for its intended use.

2. **Sound Model Validation Process**  
   This refers to the systematic evaluation of a model's performance, accuracy, and reliability. It includes monitoring, benchmarking, sensitivity analysis, and other techniques to ensure that the model is functioning as intended and meets the required standards for accuracy and robustness.

3. **Governance and Control Mechanisms**  
   This element involves establishing clear roles and responsibilities, defining the authority to restrict model usage, and implementing effective oversight and control mechanisms. It includes board and senior management oversight, policies and procedures, compliance, and an appropriate organizational structure to ensure that model risk is managed effectively.

These elements work together to ensure that models are developed, validated, and used in a way that minimizes risk and maximizes reliability.

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 4,409 chars retrieved</sub>

The three core elements of a comprehensive model validation framework are:  
1. Evaluation of conceptual soundness, including developmental evidence [V.1]  
2. Ongoing monitoring, including process verification and benchmarking [V.3]  
3. Outcomes analysis, including back-testing [V.4]

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 4,409 chars retrieved</sub>

The three core elements of a comprehensive model validation framework are:  
1. Evaluation of conceptual soundness, including developmental evidence [V.1]  
2. Ongoing monitoring, including process verification and benchmarking [V.3]  
3. Outcomes analysis, including back-testing [V.4]

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-05 — What does evaluation of conceptual soundness involve?

### Reference answer (your golden set)

> It involves assessing the quality of the model's design and construction, reviewing documentation and empirical evidence supporting the methods used and variables selected, and ensuring judgment used in design is well informed and consistent with published theory and sound industry practice.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 3 chunks, 8,850 chars retrieved</sub>

Evaluation of conceptual soundness involves assessing the quality of the model design and construction, including review of documentation and empirical evidence supporting the methods used and variables selected for the model. It entails ensuring that judgment exercised in model design and construction is well informed, carefully considered, and consistent with published research and sound industry practice. Developmental evidence should be reviewed before a model goes into use and also as part of the ongoing validation process, in particular whenever there is a material change in the model [V.2 §1].

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 3,750 chars retrieved</sub>

Evaluation of conceptual soundness involves assessing the quality of the model design and construction, reviewing documentation and empirical evidence supporting the methods used and variables selected for the model, and ensuring that qualitative, judgmental assessments are conducted in an appropriate and systematic manner, are well supported, and are documented [V.2].

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 3,750 chars retrieved</sub>

Evaluation of conceptual soundness involves assessing the quality of the model design and construction, reviewing documentation and empirical evidence supporting the methods used and variables selected for the model, and ensuring that qualitative, judgmental assessments are conducted in an appropriate and systematic manner, are well supported, and are documented [V.2].

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-06 — What is outcomes analysis and how is backtesting used within it?

### Reference answer (your golden set)

> Outcomes analysis compares model outputs to corresponding actual outcomes. Backtesting is one form of outcomes analysis that compares actual outcomes with model forecasts over a sample period not used in model development, typically using statistical tests to confirm results are within expected ranges.

### parentdoc

<sub>context_recall 0.6666666667 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 2 chunks, 6,980 chars retrieved</sub>

Outcomes analysis is a comparison of model outputs to corresponding actual outcomes, aimed at evaluating model performance by establishing expected ranges for those outcomes in relation to the model's objectives and assessing the reasons for observed variation. Back-testing is one form of outcomes analysis, involving the comparison of actual outcomes with model forecasts during a sample time period not used in model development and at an observation frequency matching the forecast horizon or performance window of the model. The comparison is generally done using expected ranges or statistical confidence intervals around the model forecasts. When outcomes fall outside those intervals, the bank should analyze the discrepancies and investigate the causes. The objective of the analysis is to determine whether differences stem from the omission of material factors, errors in model specification, or purely random variation. 

Back-testing is used within outcomes analysis to compare actual outcomes with model forecasts, helping to evaluate the accuracy of the model's forecasts and identify potential issues with the model's assumptions or performance.

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 5,992 chars retrieved</sub>

Outcomes analysis is a component of the validation process that involves comparing model outputs to corresponding actual outcomes. It helps evaluate model performance by establishing expected ranges for actual outcomes in relation to the model's objectives and assessing the reasons for observed variation. Backtesting is a specific form of outcomes analysis that compares actual outcomes with model forecasts during a sample time period not used in model development, typically at the same frequency as the model's forecast horizon or performance window. It uses expected ranges or statistical confidence intervals around the model forecasts. If outcomes fall outside these intervals, the bank should analyze the discrepancies and investigate the causes, such as omitted factors, specification errors, or random variation. Backtesting is not a substitute for outcomes analysis but complements it, and it is essential for assessing model performance over time.

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 5,992 chars retrieved</sub>

Outcomes analysis is a component of the validation process that involves comparing model outputs to corresponding actual outcomes. It helps evaluate model performance by establishing expected ranges for actual outcomes in relation to the model's objectives and assessing the reasons for observed variation. Backtesting is a specific form of outcomes analysis that compares actual outcomes with model forecasts during a sample time period not used in model development, typically at the same frequency as the model's forecast horizon or performance window. It uses expected ranges or statistical confidence intervals around the model forecasts. If outcomes fall outside these intervals, the bank should analyze the discrepancies and investigate the causes, such as omitted factors, specification errors, or random variation. Backtesting is not a substitute for outcomes analysis but complements it, and it is essential for assessing model performance over time.

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-07 — Why should model validation be performed independently of model development?

### Reference answer (your golden set)

> Validation should be done by people who are not responsible for development and who can provide effective challenge. Independence, combined with competence and influence, ensures that validation findings are objective and that identified issues are appropriately acted upon.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 2 chunks, 10,112 chars retrieved</sub>

Model validation should be performed independently of model development to ensure that the incentives are aligned with the goals of model validation [V. MODEL VALIDATION §V.3]. Independence helps prevent bias and ensures that the validation process is objective and free from any conflicts of interest that may arise from being involved in the development or use of the model. This allows for a more accurate assessment of the model's validity and reliability.

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 4,447 chars retrieved</sub>

Model validation should be performed independently of model development to ensure that the validation process is free from any biases or incentives that may influence the results [V] MODEL VALIDATION. This independence helps maintain objectivity and ensures that the validation is conducted with the goal of verifying the model's performance as expected, in line with its design objectives and business uses [V] MODEL VALIDATION.

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 5 chunks, 4,447 chars retrieved</sub>

Model validation should be performed independently of model development to ensure that the validation process is free from any biases or incentives that may influence the results [V] MODEL VALIDATION. This independence helps maintain objectivity and ensures that the validation is conducted with the goal of verifying the model's performance as expected, in line with its design objectives and business uses [V] MODEL VALIDATION.

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-08 — Who holds ultimate responsibility for model risk management within a bank?

### Reference answer (your golden set)

> The board of directors and senior management are ultimately responsible. They are responsible for establishing a strong model risk management framework, setting policies, and ensuring model risk is managed as an enterprise-wide function.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 4 chunks, 7,178 chars retrieved</sub>

The ultimate responsibility for model risk management within a bank is held by the model owner, who is tasked with ensuring that models are properly developed, implemented, and used. This responsibility includes ensuring that models in use have undergone appropriate validation and approval processes, and that issues identified through validation are communicated to relevant individuals and business users throughout the organization. 

[VI.3 §Model Owner]

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.75 &nbsp;|&nbsp; 5 chunks, 4,311 chars retrieved</sub>

The ultimate responsibility for model risk management within a bank is held by the model owner, who is tasked with ensuring that models are properly developed, implemented, and used [VI.3].

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.75 &nbsp;|&nbsp; 5 chunks, 4,311 chars retrieved</sub>

The ultimate responsibility for model risk management within a bank is held by the model owner, who is tasked with ensuring that models are properly developed, implemented, and used [VI.3].

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-09 — What should a bank's model inventory contain?

### Reference answer (your golden set)

> The model inventory should describe all models in use, under development, or recently retired, and record key information such as each model's purpose, products or activities it covers, inputs and outputs, restrictions on use, whether it is working as intended, and its validation status.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 1.0 &nbsp;|&nbsp; 3 chunks, 10,909 chars retrieved</sub>

The model inventory should describe the purpose and products for which the model is designed, actual or expected usage, and any restrictions on use. It is useful for the inventory to list the type and source of inputs used by a given model and underlying components (which may include other models), as well as model outputs and their intended use. It should also indicate whether models are functioning properly, provide a description of when they were last updated, and list any exceptions to policy. The inventory should include the names of individuals responsible for various aspects of the model development and validation; the dates of completed and planned validation activities; and the time frame during which the model is expected to remain valid. 

[VI.6 §V.3]

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 6,291 chars retrieved</sub>

A bank's model inventory should contain the purpose and products for which the model is designed, actual or expected usage, and any restrictions on use [VI.6 §V.3]. It should also describe the type and source of inputs used by a given model and underlying components, as well as model outputs and their intended use [VI.6 §V.3]. The inventory should indicate whether models are functioning properly, provide a description of when they were last updated, and list any exceptions to policy [VI.6 §V.3]. Additionally, it should include the names of individuals responsible for various aspects of the model development and validation, the dates of completed and planned validation activities, and the time frame during which the model is expected to remain valid [VI.6 §V.3].

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 6,291 chars retrieved</sub>

A bank's model inventory should contain the purpose and products for which the model is designed, actual or expected usage, and any restrictions on use [VI.6 §V.3]. It should also describe the type and source of inputs used by a given model and underlying components, as well as model outputs and their intended use [VI.6 §V.3]. The inventory should indicate whether models are functioning properly, provide a description of when they were last updated, and list any exceptions to policy [VI.6 §V.3]. Additionally, it should include the names of individuals responsible for various aspects of the model development and validation, the dates of completed and planned validation activities, and the time frame during which the model is expected to remain valid [VI.6 §V.3].

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-10 — How should vendor and other third-party models be validated?

### Reference answer (your golden set)

> Banks are responsible for validating vendor and third-party products just as they would internally developed models. They should obtain appropriate documentation, understand the product's capabilities and limitations, perform ongoing monitoring and outcomes analysis, and have contingency plans for when the product is no longer available or supported.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 3 chunks, 7,568 chars retrieved</sub>

Vendor and other third-party models should be validated following the same principles as in-house models, with additional considerations specific to external models. Banks should ensure that vendors provide developmental evidence explaining the product components, design, and intended use, and appropriate testing results that show the product works as expected. Vendors should clearly indicate the model's limitations and assumptions and where the product's use may be problematic. Banks should expect vendors to conduct ongoing performance monitoring and outcomes analysis, with disclosure to their clients, and to make appropriate modifications and updates over time. Additionally, banks should validate their own use of vendor products, relying on sensitivity analysis and benchmarking when necessary, and customizing vendor models to their specific circumstances while documenting and justifying these customization choices. Banks should also investigate the relevance of input data or assumptions provided by vendors and conduct ongoing monitoring and outcomes analysis of vendor model performance using their own outcomes. 

[VR11-7 §V.5]

### recursive

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 6,185 chars retrieved</sub>

Vendor and other third-party models should be validated following the same principles as in-house models, with additional considerations for external components and proprietary aspects [V.5]. Banks should ensure that vendors provide developmental evidence explaining the product components, design, and intended use, and appropriate testing results that show the product works as expected. Vendors should also clearly indicate the model's limitations and assumptions and where the product's use may be problematic. Banks should expect vendors to conduct ongoing performance monitoring and outcomes analysis, with disclosure to their clients, and to make appropriate modifications and updates over time [V.5].

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 6,185 chars retrieved</sub>

Vendor and other third-party models should be validated following the same principles as in-house models, with additional considerations for external components and proprietary aspects [V.5]. Banks should ensure that vendors provide developmental evidence explaining the product components, design, and intended use, and appropriate testing results that show the product works as expected. Vendors should also clearly indicate the model's limitations and assumptions and where the product's use may be problematic. Banks should expect vendors to conduct ongoing performance monitoring and outcomes analysis, with disclosure to their clients, and to make appropriate modifications and updates over time [V.5].

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---

## sr-11 — How does 'effective challenge' apply across both model development and model validation?

### Reference answer (your golden set)

> Effective challenge appears throughout the model lifecycle: during development and use it means critically reviewing design, assumptions, and appropriateness of the model for its purpose, and during validation it is delivered by independent, competent, and influential parties who identify limitations and drive corrective changes. It is the unifying discipline linking development, use, and validation.

### parentdoc

<sub>context_recall 1.0 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 4 chunks, 18,686 chars retrieved</sub>

Effective challenge applies across both model development and model validation through the involvement of objective, informed parties who can identify model limitations and assumptions and produce appropriate changes. This is supported by incentives, competence, and influence, which are essential for ensuring that challenges are effective. In model development, effective challenge may involve staff who are not responsible for development or use, ensuring independence and alignment of incentives with validation goals. In model validation, effective challenge involves critical review by an independent party to ensure proper validation, which is subject to additional activities to confirm validity. The role of effective challenge is to identify and address model limitations, thereby reducing model risk.

### recursive

<sub>context_recall 0.6666666667 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 5,222 chars retrieved</sub>

Effective challenge applies across both model development and model validation by ensuring that critical analysis is conducted by objective, informed parties who can identify model limitations and assumptions. This includes the development process, where competence and incentives are necessary to conduct appropriate analysis and critique, and during validation, where the rigor and sophistication of validation should be commensurate with the bank's overall use of models, the complexity and materiality of its models, and the size and complexity of its operations. Effective challenge also depends on the influence to ensure that actions are taken to address model issues, which comes from a combination of explicit authority, stature within the organization, and commitment and support from higher levels of management. [SR11-7 §V.3]

### semantic — same retrieval as `recursive`, identical answer

<sub>context_recall 0.6666666667 &nbsp;|&nbsp; faithfulness 0.0 &nbsp;|&nbsp; 5 chunks, 5,222 chars retrieved</sub>

Effective challenge applies across both model development and model validation by ensuring that critical analysis is conducted by objective, informed parties who can identify model limitations and assumptions. This includes the development process, where competence and incentives are necessary to conduct appropriate analysis and critique, and during validation, where the rigor and sophistication of validation should be commensurate with the bank's overall use of models, the complexity and materiality of its models, and the size and complexity of its operations. Effective challenge also depends on the influence to ensure that actions are taken to address model issues, which comes from a combination of explicit authority, stature within the organization, and commitment and support from higher levels of management. [SR11-7 §V.3]

### My verdict

| | correct | complete | padded | actionable |
|---|---|---|---|---|
| parentdoc |  |  |  |  |
| recursive |  |  |  |  |

Best: ______   Notes:

---
