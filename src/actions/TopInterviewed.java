package actions;

import java.util.HashSet;
import java.util.Set;

public class TopInterviewed {
    public int reverse(int x) {
        int result = 0;

        while (x != 0) {
            int tail = x % 10;
            int newResult = result * 10 + tail;
            if ((newResult - tail) / 10 != result) {
                return 0;
            }
            result = newResult;
            x = x / 10;
        }

        return result;
    }

    public int myAtoi(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }
        int sign = 1, base = 0, i = 0;
        while (i < str.length() && str.charAt(i) == ' ') {
            i++;
        }
        if (i == str.length())
            return 0;
        if (str.charAt(i) == '-' || str.charAt(i) == '+') {
            if (str.charAt(i) == '-') {
                sign = -1;
            }
            i++;
        }
        while (i < str.length() && str.charAt(i) >= '0' && str.charAt(i) <= '9') {
            if (base > Integer.MAX_VALUE / 10 || (base == Integer.MAX_VALUE / 10 && str.charAt(i) - '0' > 7)) {
                if (sign == 1) return Integer.MAX_VALUE;
                else return Integer.MIN_VALUE;
            }
            base = 10 * base + (str.charAt(i) - '0');
            i++;
        }
        return base * sign;
    }

    public int romanToInt(String s) {
        int nums[] = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            switch (s.charAt(i)) {
                case 'M':
                    nums[i] = 1000;
                    break;
                case 'D':
                    nums[i] = 500;
                    break;
                case 'C':
                    nums[i] = 100;
                    break;
                case 'L':
                    nums[i] = 50;
                    break;
                case 'X':
                    nums[i] = 10;
                    break;
                case 'V':
                    nums[i] = 5;
                    break;
                case 'I':
                    nums[i] = 1;
                    break;
            }
        }
        int sum = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] < nums[i + 1])
                sum -= nums[i];
            else
                sum += nums[i];
        }
        return sum + nums[nums.length - 1];
    }

    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int index = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                index++;
                nums[index] = nums[i];
            }
        }
        return index + 1;

    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public int strStr(String haystack, String needle) {
        if (needle == null || needle.length() == 0)
            return 0;
        if (haystack == null || haystack.length() == 0)
            return -1;

        for (int i = 0; i < haystack.length(); i++) {
            if (i + needle.length() > haystack.length())
                break;
            for (int j = 0; j < needle.length(); j++) {
                if (haystack.charAt(i + j) != needle.charAt(j))
                    break;
                if (j == needle.length() - 1)
                    return i;
            }
        }
        return -1;
    }

    public boolean isValidSudoku(char[][] board) {
        Set seen = new HashSet();
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char number = board[i][j];
                if (number != '.') {
                    if (!seen.add(number + " in row " + i) ||
                            !seen.add(number + " in column " + j) ||
                            !seen.add(number + " in block " + i / 3 + "-" + j / 3))
                        return false;
                }
            }
        }
        return true;
    }

    public boolean isMatch(String s, String p) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < p.length(); i++) {
            if (p.charAt(i) == '*' && i > 0 && p.charAt(i - 1) == '*')
                continue;
            sb.append(p.charAt(i));
        }
        p = sb.toString();
        int[][] dp = new int[s.length()][p.length()];
        return isMatchRecur(s, p, 0, 0, dp);
    }

    private boolean isMatchRecur(String s, String p, int sIndex, int pIndex, int[][] dp) {
        if (sIndex == s.length()) {
            while (pIndex < p.length()) {
                if (p.charAt(pIndex) != '*')
                    return false;
                pIndex++;
            }
            return true;
        }
        if (pIndex == p.length()) {
            return false;
        }

        if (dp[sIndex][pIndex] != 0)
            return dp[sIndex][pIndex] == 1;

        if (s.charAt(sIndex) == p.charAt(pIndex) || p.charAt(pIndex) == '?') {
            if (isMatchRecur(s, p, sIndex + 1, pIndex + 1, dp))
                dp[sIndex][pIndex] = 1;

        } else if (p.charAt(pIndex) == '*') {
            if (isMatchRecur(s, p, sIndex + 1, pIndex, dp) || isMatchRecur(s, p, sIndex, pIndex + 1, dp))
                dp[sIndex][pIndex] = 1;
        } else {
            dp[sIndex][pIndex] = 0;
        }
        return dp[sIndex][pIndex] == 1;
    }

    public boolean isMatchDP(String s, String p) {
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[s.length()][p.length()] = true;
        for (int i = p.length() - 1; i >= 0; i--) {
            if (p.charAt(i) != '*')
                break;
            else
                dp[s.length()][i] = true;
        }
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = p.length() - 1; j >= 0; j--) {
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')
                    dp[i][j] = dp[i + 1][j + 1];
                else if (p.charAt(j) == '*')
                    dp[i][j] = dp[i + 1][j] || dp[i][j + 1];
                else
                    dp[i][j] = false;
            }
        }
        return dp[0][0];
    }

    double myPow(double x, int n) {
        if (n == 0)
            return 1;
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        double ans = 1;
        while (n > 0) {
            if (n % 2 == 1)
                ans *= x;
            x *= x;
            n /= 2;
        }
        return ans;
    }

    public int[] plusOne(int[] digits) {
        int carry = 1;
        for (int i = digits.length - 1; i >= 0; i--) {
            int sum = digits[i] + carry;
            if (sum == 10) {
                digits[i] = 0;
                carry = 1;
            } else {
                digits[i] = sum;
                return digits;
            }
        }

        int[] newNumber = new int[digits.length + 1];
        newNumber[0] = 1;
        return newNumber;
    }

    public int mySqrt(int x) {
        if (x == 0)
            return 0;
        int left = 1, right = Integer.MAX_VALUE;
        while (true) {
            int mid = left + (right - left) / 2;
            if (mid > x / mid) {
                right = mid - 1;
            } else {
                if (mid + 1 > x / (mid + 1))
                    return mid;
                left = mid + 1;
            }
        }
    }
}
