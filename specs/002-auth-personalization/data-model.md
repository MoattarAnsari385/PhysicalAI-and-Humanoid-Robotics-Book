# Data Model: Authentication + Personalization + Urdu Translation

**Feature**: `002-auth-personalization`
**Created**: 2025-12-10
**Status**: Complete

## Database Schema

### users Table

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, NOT NULL | Unique identifier for the user |
| email | VARCHAR(255) | UNIQUE, NOT NULL | User's email address |
| password_hash | VARCHAR(255) | NOT NULL | Hashed password using secure algorithm |
| software_level | VARCHAR(20) | CHECK: 'beginner', 'intermediate', 'advanced' | User's software experience level |
| gpu_type | VARCHAR(100) | NULLABLE | Type of GPU user has access to |
| jetson_model | VARCHAR(100) | NULLABLE | Jetson device model user owns |
| created_at | TIMESTAMP | NOT NULL, DEFAULT: NOW() | Account creation timestamp |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT: NOW() | Last update timestamp |

### user_preferences Table

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| user_id | UUID | PRIMARY KEY, FOREIGN KEY (users.id), NOT NULL | Reference to user |
| level_preference | VARCHAR(20) | CHECK: 'beginner', 'intermediate', 'advanced' | Preferred content complexity level |
| language | VARCHAR(5) | CHECK: 'en', 'ur', DEFAULT: 'en' | Preferred language (English/Urdu) |
| created_at | TIMESTAMP | NOT NULL, DEFAULT: NOW() | Preference creation timestamp |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT: NOW() | Last preference update timestamp |

## Entity Relationships

```
users (1) <---> (1) user_preferences
```

- Each user has exactly one set of preferences
- User preferences are linked to user via foreign key
- When a user is deleted, their preferences should also be deleted (CASCADE)

## Validation Rules

### users Table
- Email must be a valid email format
- Password must meet security requirements (min 8 characters, complexity)
- software_level must be one of: 'beginner', 'intermediate', 'advanced'
- gpu_type and jetson_model are optional fields
- Email uniqueness is enforced at database level

### user_preferences Table
- user_id must reference an existing user
- level_preference must be one of: 'beginner', 'intermediate', 'advanced'
- language must be one of: 'en', 'ur'
- Each user can only have one set of preferences (enforced by primary key)

## Indexes

### Required Indexes
- `users.email`: UNIQUE INDEX for fast email lookups during authentication
- `user_preferences.user_id`: INDEX for fast preference retrieval by user

### Optional Indexes
- `users.created_at`: INDEX for user analytics and reporting
- `user_preferences.updated_at`: INDEX for preference usage analytics

## State Transitions

### User Registration Process
1. User provides email, password, and background information
2. System validates input data
3. System creates user record in `users` table
4. System creates default preferences in `user_preferences` table
5. System sends verification email (optional)

### Preference Update Process
1. User changes personalization or language preference
2. System validates new preference values
3. System updates `user_preferences` record
4. System broadcasts preference change to UI components
5. System optionally syncs with client-side cache

## Data Lifecycle

### Creation
- User account created during signup process
- Default preferences created automatically with account
- Background information captured during signup

### Updates
- User can update preferences at any time
- System updates `updated_at` timestamp automatically
- Changes are reflected in UI immediately

### Deletion
- User account deletion triggers preference deletion (CASCADE)
- All personalization data removed when account is deleted
- No orphaned preference records allowed

## Security Considerations

### Data Protection
- Passwords stored as bcrypt hashes (never plain text)
- Sensitive user data encrypted at rest
- Database connections use SSL encryption
- Audit logging for sensitive operations

### Access Control
- User data only accessible to the owning user
- Admin access limited and audited
- API endpoints validate user ownership
- No direct database access from client-side code

## Performance Considerations

### Query Optimization
- Frequently accessed user data cached in application layer
- Preference data loaded with user session initialization
- Database queries use proper indexing
- Connection pooling for database operations

### Scalability
- UUID primary keys allow for distributed systems
- Read replicas for preference queries
- Database sharding strategy for large user base
- Efficient pagination for user management