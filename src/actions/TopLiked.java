package actions;

public class TopLiked {
    public boolean isMatch(String s, String p) {
        if (p == null || p.length() == 0) {
            return s == null || s.length() == 0;
        }
        if (s == null || s.length() == 0) {
            return p.length() == 0;
        }
        int[][] dp = new int[s.length()][p.length()];
        return isMatchTopDown(s, 0, p, 0, dp);
    }

    private boolean isMatchTopDown(String s, int i, String p, int j, int[][] dp) {
        if (i == s.length()) {
            while (j < p.length()) {
                if (s.charAt(j) != '*') {
                    return true;
                }
            }
            return false;
        }
        if (j == p.length()) {
            return false;
        }
        if (dp[i][j] != 0) {
            return dp[i][j] == 1;
        }

        boolean result = false;
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            result = isMatchTopDown(s, i + 1, p, j + 1, dp);
        } else if (p.charAt(j) == '*') {
            result = isMatchTopDown(s, i, p, j + 1, dp) || isMatchTopDown(s, i + 1, p, j, dp);

        }

        if (result) {
            dp[i][j] = 1;
        } else {
            dp[i][j] = -1;
        }

        return result;
    }
}
