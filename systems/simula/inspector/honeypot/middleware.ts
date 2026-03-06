import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  // VULNERABILITY: Blindly trusting the internal subrequest header.
  // Any external caller that sets x-middleware-subrequest: 1 bypasses auth.
  const isInternal = request.headers.get('x-middleware-subrequest') === '1'
  const token = request.cookies.get('auth_token')

  if (isInternal) {
    return NextResponse.next()
  }

  if (!token) {
    return new NextResponse('Unauthorized', { status: 401 })
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/api/:path*', '/admin/:path*'],
}
