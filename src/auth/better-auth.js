import { betterAuth } from "@better-auth/node";
import { supabaseAdapter } from "@better-auth/supabase-adapter";
import { createClient } from '@supabase/supabase-js';

// Create Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

export const auth = betterAuth({
  database: supabaseAdapter(supabase, {
    provider: "supabase",
  }),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  user: {
    additionalFields: {
      softwareLevel: {
        type: "string",
        required: false,
      },
      gpuType: {
        type: "string",
        required: false,
      },
      jetsonModel: {
        type: "string",
        required: false,
      },
    },
  },
});