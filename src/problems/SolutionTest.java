package problems;

import org.junit.runner.RunWith;

import java.util.List;

import static org.junit.Assert.*;

public class SolutionTest {
    Solution s = new Solution();

    @org.junit.Test
    public void testMethod() {
        s.threeSumClosest(new int[]{-1, 2, 1, -4}, 1);
        assertTrue(s.isPalindrome(121));
    }

    @org.junit.Test
    public void testSubstring(){
        String str = "barfoothefoobarman";
        String[] words = {"foo","bar"};
        s.findSubstring(str, words);
    }
}
