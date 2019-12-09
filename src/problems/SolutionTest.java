package problems;

import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

public class SolutionTest {
    Solution s = new Solution();

    @org.junit.Test
    public void testMethod() {
        s.nextPermutation(new int[]{1,5,1});
        s.threeSumClosest(new int[]{-1, 2, 1, -4}, 1);
        assertTrue(s.isPalindrome(121));
    }

    @org.junit.Test
    public void testSubstring(){
        String str = "barfoothefoobarman";
        String[] words = {"foo","bar"};
        s.findSubstring(str, words);
    }

    @org.junit.Test
    public void testReverse(){
        Integer[] nums = {1,2,3};
        Collections.reverse(Arrays.asList(nums));
        System.out.println(Arrays.asList(nums));
    }

    @org.junit.Test
    public void testHanoi(){
        s.hanoi(3);
    }
}
