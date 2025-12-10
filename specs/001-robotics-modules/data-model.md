# Data Model: Physical AI & Humanoid Robotics Textbook

## Core Entities

### Module
- **id**: string (unique identifier, e.g., "module-1-ros2")
- **title**: string (display title)
- **description**: string (brief overview)
- **learningOutcomes**: array of strings (measurable objectives)
- **sections**: array of Section references
- **wordCount**: integer (target: 6,000-8,000 words)
- **prerequisites**: array of strings (required knowledge)
- **deliverables**: array of Deliverable references

### Section
- **id**: string (unique within module)
- **title**: string (display title)
- **content**: string (markdown content, target: 800-1,200 words)
- **module**: Module reference
- **order**: integer (sequence within module)
- **figures**: array of Figure references
- **codeExamples**: array of CodeExample references
- **assessments**: array of Assessment references

### Figure
- **id**: string (unique identifier)
- **title**: string (descriptive title)
- **description**: string (what the figure illustrates)
- **filePath**: string (path to image file)
- **format**: string (svg, png, jpeg)
- **caption**: string (explanation of learning takeaway)
- **altText**: string (for accessibility)
- **section**: Section reference

### CodeExample
- **id**: string (unique identifier)
- **title**: string (descriptive title)
- **language**: string (python, c++, etc.)
- **code**: string (the actual code)
- **explanation**: string (150-300 words explaining the code)
- **section**: Section reference
- **filePath**: string (path to code file if external)

### Deliverable
- **id**: string (unique identifier)
- **title**: string (descriptive title)
- **description**: string (what the deliverable accomplishes)
- **type**: string (robot_package, diagram, workflow, architecture)
- **module**: Module reference
- **filePath**: string (path to deliverable file)

### Assessment
- **id**: string (unique identifier)
- **title**: string (descriptive title)
- **type**: string (formative, summative, mini-project, capstone)
- **rubric**: object (grading criteria)
- **section**: Section reference
- **module**: Module reference

### GlossaryTerm
- **id**: string (unique identifier)
- **term**: string (the term being defined)
- **definition**: string (clear, concise definition)
- **canonicalDocRef**: string (reference to official documentation)
- **module**: Module reference (optional, if specific to a module)

## Relationships

### Module ↔ Section
- One-to-Many: A Module contains 5-6 Sections

### Section ↔ Figure
- One-to-Many: A Section can have multiple Figures (minimum 3-5)

### Section ↔ CodeExample
- One-to-Many: A Section can have multiple CodeExamples

### Section ↔ Assessment
- One-to-Many: A Section can have multiple Assessments (including formative assessments)

### Module ↔ Deliverable
- One-to-Many: A Module has specific Deliverables

### Module ↔ GlossaryTerm
- One-to-Many: A Module may introduce specific GlossaryTerms

## Validation Rules

1. **Module**:
   - wordCount must be between 6,000 and 8,000
   - must have 5-6 sections
   - must have 20-30 figures (4-6 per section)
   - learningOutcomes must be measurable

2. **Section**:
   - content word count must be 800-1,200
   - must have 3-5 figures
   - must have at least one code example

3. **Figure**:
   - must have altText for accessibility
   - caption must explain the learning takeaway
   - filePath must exist in assets/

4. **CodeExample**:
   - explanation must be 150-300 words
   - language must be specified

5. **Assessment**:
   - rubric must be defined for grading

## State Transitions

### Module states:
- **Draft**: Initial state
- **In Review**: Content being reviewed
- **Reviewed**: Review complete, awaiting approval
- **Published**: Content published and accessible

### Section states:
- **Outline**: Basic structure created
- **Draft**: Content written
- **Reviewed**: Content reviewed
- **Assessed**: Assessment items created
- **Complete**: Ready for publishing