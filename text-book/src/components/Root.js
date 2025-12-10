import React from 'react';
import { AuthProvider } from '../contexts/AuthContext';

// Root component that wraps the entire application with AuthProvider
function Root({children}) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}

export default Root;