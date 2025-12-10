import React, { useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import { useAuth } from '@better-auth/react';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useUser();
  const { session } = useAuth();

  useEffect(() => {
    // Check authentication status and redirect if not authenticated
    if (!loading && !isAuthenticated && !session) {
      window.location.href = '/login';
    }
  }, [isAuthenticated, loading, session]);

  if (loading || (!isAuthenticated && !session)) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        Loading...
      </div>
    );
  }

  return children;
};

export default ProtectedRoute;