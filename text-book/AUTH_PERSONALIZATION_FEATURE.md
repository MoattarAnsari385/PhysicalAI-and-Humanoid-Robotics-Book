# Authentication & Personalization Feature Implementation

This directory contains the implementation of the authentication, personalization, and Urdu translation features for the Physical AI & Humanoid Robotics Textbook.

## Features Implemented

### 1. Authentication System
- **Better-Auth Integration**: Secure email/password authentication
- **User Registration**: Signup form with background questions
- **User Profile**: Stores software experience level, GPU access, and Jetson device information
- **Session Management**: JWT-based authentication with secure token handling

### 2. Personalization Features
- **Content Level Toggle**: Beginner/Intermediate/Advanced content adjustment
- **User Preference Persistence**: Saves preferences to database and local storage
- **Dynamic Content**: Content adjusts based on user's experience level

### 3. Language Translation
- **English/Urdu Toggle**: Switch between English and Urdu content
- **RTL Support**: Right-to-left layout for Urdu content
- **Font Support**: Proper Urdu font rendering

## File Structure

```
text-book/
├── src/
│   ├── auth/
│   │   └── better-auth.js          # Better-Auth configuration
│   ├── contexts/
│   │   └── UserContext.js          # Global user state management
│   ├── components/
│   │   ├── LoginForm.js            # Login form component
│   │   ├── SignupForm.js           # Signup form with background questions
│   │   ├── PersonalizeToggle.js    # Personalization toggle component
│   │   ├── LanguageToggle.js       # Language toggle component
│   │   ├── ChapterControls.js      # Combined controls component
│   │   ├── PersonalizedContent.js  # Dynamic content component
│   │   └── ProtectedRoute.js       # Authentication guard
│   ├── utils/
│   │   └── api.js                  # API utility functions
│   ├── theme/
│   │   ├── App.js                  # Main app wrapper with auth context
│   │   └── ChapterLayout.js        # Custom layout for chapters
│   └── pages/
│       ├── login.js                # Login page
│       ├── signup.js               # Signup page
│       └── home.js                 # Home/Library page
├── docs/
│   └── intro.md                    # Sample chapter
└── docusaurus.config.js            # Docusaurus configuration
```

## Database Schema

The implementation uses Supabase with the following tables:

### users table
- id (UUID, primary key)
- email (VARCHAR, unique)
- password_hash (VARCHAR)
- software_level (VARCHAR: beginner/intermediate/advanced)
- gpu_type (VARCHAR, optional)
- jetson_model (VARCHAR, optional)
- created_at, updated_at (timestamps)

### user_preferences table
- user_id (UUID, foreign key to users.id)
- level_preference (VARCHAR: beginner/intermediate/advanced)
- language (VARCHAR: en/ur)
- created_at, updated_at (timestamps)

## Environment Variables

Add these to your `.env` file:

```env
BETTER_AUTH_URL=your_auth_url
BETTER_AUTH_SECRET=your_secret_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DATABASE_URL=your_database_connection_string
```

## How to Use

### For Users
1. Visit `/signup` to create an account with background information
2. Use `/login` to sign in
3. Navigate to any chapter to see personalization controls
4. Use the toggle buttons to adjust content level and language

### For Content Authors
1. Tag content sections with appropriate difficulty levels
2. Provide Urdu translations for content sections
3. Use the `ChapterControls` component in chapter layouts

## Technical Implementation Details

### Authentication Flow
1. User registers via signup form
2. Background information is stored in user profile
3. User authenticates using Better-Auth
4. Session is managed via JWT tokens
5. User is redirected to Home/Library after login

### Personalization System
1. User preferences are loaded from database on authentication
2. Personalization controls update both local state and database
3. Content components listen for preference changes
4. Appropriate content sections are displayed based on user level

### Language Translation
1. Language toggle updates both local state and database
2. RTL layout is applied when Urdu is selected
3. Content switches between English and Urdu versions
4. Font rendering is optimized for Urdu text

## Testing the Implementation

1. Start the Docusaurus development server: `npm run start`
2. Navigate to `/signup` to create a new account
3. Fill in background information during registration
4. Login at `/login` and verify redirect to Home
5. Visit a chapter page and test personalization toggles
6. Verify content adjusts based on selected preferences
7. Test language toggle and RTL layout switching
8. Confirm preferences persist across sessions

## Security Considerations

- JWT tokens are stored securely using httpOnly cookies
- Passwords are hashed using secure algorithms
- Database connections use SSL encryption
- Input validation is performed on all user data
- Authentication state is properly managed

## Performance Optimizations

- JWT tokens are cached locally to reduce authentication calls
- Database queries are optimized with proper indexing
- Authentication state is efficiently managed in React Context
- Translation dictionaries are loaded once and cached
- Content switching is instantaneous after initial load