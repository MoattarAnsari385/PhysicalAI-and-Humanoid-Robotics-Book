// Client-side Better-Auth configuration for Docusaurus
import { createAuthClient } from "better-auth/client";

// Create the auth client
export const { signIn, signUp, signOut, useSession } = createAuthClient({
  baseURL: typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000',
  fetchOptions: {
    // Add any additional fetch options here if needed
  }
});