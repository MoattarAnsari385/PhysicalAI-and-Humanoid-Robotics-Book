import { betterAuth } from "better-auth";

// Create the Better-Auth instance with extended user schema
// Using memory adapter for development compatibility
export const auth = betterAuth({
  database: {
    provider: "memory", // Use memory adapter for now
  },
  secret: process.env.BETTER_AUTH_SECRET || "your-secret-key-change-this-for-production",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  user: {
    // Extend the default user schema with additional fields
    additionalFields: {
      programmingExperience: {
        type: "string",
        required: false,
      },
      primaryLanguages: {
        type: "string", // Store as JSON string
        required: false,
      },
      hasNvidiaGPU: {
        type: "boolean",
        required: false,
      },
      ownsDevices: {
        type: "string", // Store as JSON string
        required: false,
      },
      experienceLevel: {
        type: "string",
        required: false,
      },
      avatarColor: {
        type: "string",
        required: false,
      },
      theme: {
        type: "string",
        required: false,
        defaultValue: "dark",
      },
    },
  },
});